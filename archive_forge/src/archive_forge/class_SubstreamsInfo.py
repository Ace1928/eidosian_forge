import functools
import io
import operator
import os
import struct
from binascii import unhexlify
from functools import reduce
from io import BytesIO
from operator import and_, or_
from struct import pack, unpack
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from py7zr.compressor import SevenZipCompressor, SevenZipDecompressor
from py7zr.exceptions import Bad7zFile
from py7zr.helpers import ArchiveTimestamp, calculate_crc32
from py7zr.properties import DEFAULT_FILTERS, MAGIC_7Z, PROPERTY
class SubstreamsInfo:
    """defines the substreams of a folder"""
    __slots__ = ['digests', 'digestsdefined', 'unpacksizes', 'num_unpackstreams_folders']

    def __init__(self):
        self.digests: List[int] = []
        self.digestsdefined: List[bool] = []
        self.unpacksizes: Optional[List[int]] = None
        self.num_unpackstreams_folders: List[int] = []

    @classmethod
    def retrieve(cls, file: BinaryIO, numfolders: int, folders: List[Folder]):
        obj = cls()
        obj._read(file, numfolders, folders)
        return obj

    def _read(self, file: BinaryIO, numfolders: int, folders: List[Folder]):
        pid = file.read(1)
        if pid == PROPERTY.NUM_UNPACK_STREAM:
            self.num_unpackstreams_folders = [read_uint64(file) for _ in range(numfolders)]
            pid = file.read(1)
        else:
            self.num_unpackstreams_folders = [1] * numfolders
        if pid == PROPERTY.SIZE:
            self.unpacksizes = []
            for i in range(len(self.num_unpackstreams_folders)):
                totalsize = 0
                for j in range(1, self.num_unpackstreams_folders[i]):
                    size = read_uint64(file)
                    self.unpacksizes.append(size)
                    totalsize += size
                self.unpacksizes.append(folders[i].get_unpack_size() - totalsize)
            pid = file.read(1)
        num_digests = 0
        num_digests_total = 0
        for i in range(numfolders):
            numsubstreams = self.num_unpackstreams_folders[i]
            if numsubstreams != 1 or not folders[i].digestdefined:
                num_digests += numsubstreams
            num_digests_total += numsubstreams
        if pid == PROPERTY.CRC:
            defined = read_boolean(file, num_digests, checkall=True)
            crcs = read_crcs(file, num_digests)
            didx = 0
            for i in range(numfolders):
                folder = folders[i]
                numsubstreams = self.num_unpackstreams_folders[i]
                if numsubstreams == 1 and folder.digestdefined and (folder.crc is not None):
                    self.digestsdefined.append(True)
                    self.digests.append(folder.crc)
                else:
                    for j in range(numsubstreams):
                        self.digestsdefined.append(defined[didx])
                        self.digests.append(crcs[didx])
                        didx += 1
            pid = file.read(1)
        if pid != PROPERTY.END:
            raise Bad7zFile('end id expected but %r found' % pid)
        if not self.digestsdefined:
            self.digestsdefined = [False] * num_digests_total
            self.digests = [0] * num_digests_total

    def write(self, file: Union[BinaryIO, WriteWithCrc]):
        if len(self.num_unpackstreams_folders) == 0:
            return
        write_byte(file, PROPERTY.SUBSTREAMS_INFO)
        solid = functools.reduce(lambda x, y: x or y != 1, self.num_unpackstreams_folders, False)
        if solid:
            write_byte(file, PROPERTY.NUM_UNPACK_STREAM)
            for n in self.num_unpackstreams_folders:
                write_uint64(file, n)
        has_multi = functools.reduce(lambda x, y: x or y > 1, self.num_unpackstreams_folders, False)
        if has_multi:
            assert self.unpacksizes
            write_byte(file, PROPERTY.SIZE)
            idx = 0
            for i, num in enumerate(self.num_unpackstreams_folders):
                for j in range(num):
                    if j + 1 != num:
                        write_uint64(file, self.unpacksizes[idx])
                    idx += 1
        if functools.reduce(lambda x, y: x or y, self.digestsdefined, False):
            write_byte(file, PROPERTY.CRC)
            write_boolean(file, self.digestsdefined, all_defined=True)
            write_crcs(file, self.digests)
        write_byte(file, PROPERTY.END)