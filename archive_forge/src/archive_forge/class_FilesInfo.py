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
class FilesInfo:
    """holds file properties"""
    __slots__ = ['files', 'emptyfiles', 'antifiles']

    def __init__(self):
        self.files: List[Dict[str, Any]] = []
        self.emptyfiles: List[bool] = []
        self.antifiles = None

    @classmethod
    def retrieve(cls, file: BinaryIO):
        obj = cls()
        obj._read(file)
        return obj

    def _read(self, fp: BinaryIO):
        numfiles = read_uint64(fp)
        self.files = [{'emptystream': False} for _ in range(numfiles)]
        numemptystreams = 0
        while True:
            prop = fp.read(1)
            if prop == PROPERTY.END:
                break
            size = read_uint64(fp)
            if prop == PROPERTY.DUMMY:
                fp.seek(size, os.SEEK_CUR)
                continue
            buffer = io.BytesIO(fp.read(size))
            if prop == PROPERTY.EMPTY_STREAM:
                isempty = read_boolean(buffer, numfiles, checkall=False)
                list(map(lambda x, y: x.update({'emptystream': y}), self.files, isempty))
                numemptystreams += isempty.count(True)
            elif prop == PROPERTY.EMPTY_FILE:
                self.emptyfiles = read_boolean(buffer, numemptystreams, checkall=False)
            elif prop == PROPERTY.NAME:
                external = buffer.read(1)
                if external == b'\x00':
                    self._read_name(buffer)
                else:
                    dataindex = read_uint64(buffer)
                    current_pos = fp.tell()
                    fp.seek(dataindex, 0)
                    self._read_name(fp)
                    fp.seek(current_pos, 0)
            elif prop == PROPERTY.CREATION_TIME:
                self._read_times(buffer, 'creationtime')
            elif prop == PROPERTY.LAST_ACCESS_TIME:
                self._read_times(buffer, 'lastaccesstime')
            elif prop == PROPERTY.LAST_WRITE_TIME:
                self._read_times(buffer, 'lastwritetime')
            elif prop == PROPERTY.ATTRIBUTES:
                defined = read_boolean(buffer, numfiles, checkall=True)
                external = buffer.read(1)
                if external == b'\x00':
                    self._read_attributes(buffer, defined)
                else:
                    dataindex = read_uint64(buffer)
                    current_pos = fp.tell()
                    fp.seek(dataindex, 0)
                    self._read_attributes(fp, defined)
                    fp.seek(current_pos, 0)
            elif prop == PROPERTY.START_POS:
                self._read_start_pos(buffer)
            else:
                raise Bad7zFile('invalid type %r' % prop)

    def _read_name(self, buffer: BinaryIO) -> None:
        for f in self.files:
            f['filename'] = read_utf16(buffer).replace('\\', '/')

    def _read_attributes(self, buffer: BinaryIO, defined: List[bool]) -> None:
        for idx, f in enumerate(self.files):
            f['attributes'] = read_uint32(buffer)[0] if defined[idx] else None

    def _read_times(self, fp: BinaryIO, name: str) -> None:
        defined = read_boolean(fp, len(self.files), checkall=True)
        external = fp.read(1)
        assert external == b'\x00'
        for i, f in enumerate(self.files):
            f[name] = ArchiveTimestamp(read_real_uint64(fp)[0]) if defined[i] else None

    def _read_start_pos(self, fp: BinaryIO) -> None:
        defined = read_boolean(fp, len(self.files), checkall=True)
        external = fp.read(1)
        assert external == 0
        for i, f in enumerate(self.files):
            f['startpos'] = read_real_uint64(fp)[0] if defined[i] else None

    def _write_times(self, fp: Union[BinaryIO, WriteWithCrc], propid, name: str) -> None:
        write_byte(fp, propid)
        defined = []
        num_defined = 0
        for f in self.files:
            if name in f.keys():
                if f[name] is not None:
                    defined.append(True)
                    num_defined += 1
                    continue
            defined.append(False)
        size = num_defined * 8 + 2
        if not reduce(and_, defined, True):
            size += bits_to_bytes(num_defined)
        write_uint64(fp, size)
        write_boolean(fp, defined, all_defined=True)
        write_byte(fp, b'\x00')
        for i, file in enumerate(self.files):
            if defined[i]:
                write_real_uint64(fp, file[name])
            else:
                pass

    def _write_prop_bool_vector(self, fp: Union[BinaryIO, WriteWithCrc], propid, vector) -> None:
        write_byte(fp, propid)
        write_boolean(fp, vector, all_defined=False)

    @staticmethod
    def _are_there(vector) -> bool:
        if vector is not None:
            if functools.reduce(or_, vector, False):
                return True
        return False

    def _write_names(self, file: Union[BinaryIO, WriteWithCrc]):
        name_defined = 0
        names = []
        name_size = 0
        for f in self.files:
            if f.get('filename', None) is not None:
                name_defined += 1
                names.append(f['filename'])
                name_size += len(f['filename'].encode('utf-16LE')) + 2
        if name_defined > 0:
            write_byte(file, PROPERTY.NAME)
            write_uint64(file, name_size + 1)
            write_byte(file, b'\x00')
            for n in names:
                write_utf16(file, n)

    def _write_attributes(self, file):
        defined = []
        num_defined = 0
        for f in self.files:
            if 'attributes' in f.keys() and f['attributes'] is not None:
                defined.append(True)
                num_defined += 1
            else:
                defined.append(False)
        size = num_defined * 4 + 2
        if num_defined != len(defined):
            size += bits_to_bytes(num_defined)
        write_byte(file, PROPERTY.ATTRIBUTES)
        write_uint64(file, size)
        write_boolean(file, defined, all_defined=True)
        write_byte(file, b'\x00')
        for i, f in enumerate(self.files):
            if defined[i]:
                write_uint32(file, f['attributes'])

    def write(self, file: Union[BinaryIO, WriteWithCrc]):
        assert self.files is not None
        write_byte(file, PROPERTY.FILES_INFO)
        numfiles = len(self.files)
        write_uint64(file, numfiles)
        emptystreams = []
        for f in self.files:
            emptystreams.append(f['emptystream'])
        if self._are_there(emptystreams):
            write_byte(file, PROPERTY.EMPTY_STREAM)
            write_uint64(file, bits_to_bytes(numfiles))
            write_boolean(file, emptystreams, all_defined=False)
        elif self._are_there(self.emptyfiles):
            self._write_prop_bool_vector(file, PROPERTY.EMPTY_FILE, self.emptyfiles)
        pos = file.tell()
        padlen = -pos & 3
        if 2 >= padlen > 0:
            padlen += 4
        if padlen > 2:
            write_byte(file, PROPERTY.DUMMY)
            write_byte(file, (padlen - 2).to_bytes(1, 'little'))
            write_bytes(file, bytes(padlen - 2))
        self._write_names(file)
        self._write_times(file, PROPERTY.LAST_WRITE_TIME, 'lastwritetime')
        self._write_attributes(file)
        write_byte(file, PROPERTY.END)