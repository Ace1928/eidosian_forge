import collections.abc
import contextlib
import datetime
import errno
import functools
import io
import os
import pathlib
import queue
import re
import stat
import sys
import time
from multiprocessing import Process
from threading import Thread
from typing import IO, Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Type, Union
import multivolumefile
from py7zr.archiveinfo import Folder, Header, SignatureHeader
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods, get_methods_names
from py7zr.exceptions import (
from py7zr.helpers import (
from py7zr.properties import DEFAULT_FILTERS, FILTER_DEFLATE64, MAGIC_7Z, get_default_blocksize, get_memory_limit
def _real_get_contents(self, password) -> None:
    if not self._check_7zfile(self.fp):
        raise Bad7zFile('not a 7z file')
    self.sig_header = SignatureHeader.retrieve(self.fp)
    self.afterheader: int = self.fp.tell()
    self.fp.seek(self.sig_header.nextheaderofs, os.SEEK_CUR)
    buffer = io.BytesIO(self.fp.read(self.sig_header.nextheadersize))
    if self.sig_header.nextheadercrc != calculate_crc32(buffer.getvalue()):
        raise Bad7zFile('invalid header data')
    header = Header.retrieve(self.fp, buffer, self.afterheader, password)
    if header is None:
        return
    header._initilized = True
    self.header = header
    header.size += 32 + self.sig_header.nextheadersize
    buffer.close()
    self.files = ArchiveFileList()
    if getattr(self.header, 'files_info', None) is None:
        return
    if hasattr(self.header, 'main_streams') and self.header.main_streams is not None:
        folders = self.header.main_streams.unpackinfo.folders
        for folder in folders:
            folder.password = password
        packinfo = self.header.main_streams.packinfo
        packsizes = packinfo.packsizes
        subinfo = self.header.main_streams.substreamsinfo
        if subinfo is not None and subinfo.unpacksizes is not None:
            unpacksizes = subinfo.unpacksizes
        else:
            unpacksizes = [x.unpacksizes[-1] for x in folders]
    else:
        subinfo = None
        folders = None
        packinfo = None
        packsizes = []
        unpacksizes = [0]
    pstat = self.ParseStatus()
    pstat.src_pos = self.afterheader
    file_in_solid = 0
    for file_id, file_info in enumerate(self.header.files_info.files):
        if not file_info['emptystream'] and folders is not None:
            folder = folders[pstat.folder]
            numinstreams = max([coder.get('numinstreams', 1) for coder in folder.coders])
            maxsize, compressed, uncompressed, packsize, solid = self._get_fileinfo_sizes(pstat, subinfo, packinfo, folder, packsizes, unpacksizes, file_in_solid, numinstreams)
            pstat.input += 1
            folder.solid = solid
            file_info['folder'] = folder
            file_info['maxsize'] = maxsize
            file_info['compressed'] = compressed
            file_info['uncompressed'] = uncompressed
            file_info['packsizes'] = packsize
            if subinfo.digestsdefined[pstat.outstreams]:
                file_info['digest'] = subinfo.digests[pstat.outstreams]
            if folder is None:
                pstat.src_pos += file_info['compressed']
            else:
                if folder.solid:
                    file_in_solid += 1
                pstat.outstreams += 1
                if folder.files is None:
                    folder.files = ArchiveFileList(offset=file_id)
                folder.files.append(file_info)
                if pstat.input >= subinfo.num_unpackstreams_folders[pstat.folder]:
                    file_in_solid = 0
                    pstat.src_pos += sum(packinfo.packsizes[pstat.stream:pstat.stream + numinstreams])
                    pstat.folder += 1
                    pstat.stream += numinstreams
                    pstat.input = 0
        else:
            file_info['folder'] = None
            file_info['maxsize'] = 0
            file_info['compressed'] = 0
            file_info['uncompressed'] = 0
            file_info['packsizes'] = [0]
        if 'filename' not in file_info:
            try:
                basefilename = self.filename
            except AttributeError:
                file_info['filename'] = 'contents'
            else:
                if basefilename is not None:
                    fn, ext = os.path.splitext(os.path.basename(basefilename))
                    file_info['filename'] = fn
                else:
                    file_info['filename'] = 'contents'
        self.files.append(file_info)
    if not self.password_protected and self.header.main_streams is not None:
        self.password_protected = any([SupportedMethods.needs_password(folder.coders) for folder in self.header.main_streams.unpackinfo.folders])