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
def _extract_single(self, fp: BinaryIO, files, path, src_end: int, q: Optional[queue.Queue], skip_notarget=True) -> None:
    """
        Single thread extractor that takes file lists in single 7zip folder.
        this may raise exception.
        """
    just_check: List[ArchiveFile] = []
    for f in files:
        if q is not None:
            q.put(('s', str(f.filename), str(f.compressed) if f.compressed is not None else '0'))
        fileish = self.target_filepath.get(f.id, None)
        if fileish is None:
            if not f.emptystream:
                just_check.append(f)
        else:
            self._check(fp, just_check, src_end)
            just_check = []
            fileish.parent.mkdir(parents=True, exist_ok=True)
            if not f.emptystream:
                if f.is_junction and (not isinstance(fileish, MemIO)) and (sys.platform == 'win32'):
                    with io.BytesIO() as ofp:
                        self.decompress(fp, f.folder, ofp, f.uncompressed, f.compressed, src_end, q)
                        dst: str = ofp.read().decode('utf-8')
                        if is_path_valid(fileish.parent.joinpath(dst), path):
                            if fileish.exists():
                                fileish.unlink()
                            if sys.platform == 'win32':
                                _winapi.CreateJunction(str(fileish), dst)
                        else:
                            raise Bad7zFile('Junction point out of target directory.')
                elif f.is_symlink and (not isinstance(fileish, MemIO)):
                    with io.BytesIO() as omfp:
                        self.decompress(fp, f.folder, omfp, f.uncompressed, f.compressed, src_end, q)
                        omfp.seek(0)
                        dst = omfp.read().decode('utf-8')
                        if is_path_valid(fileish.parent.joinpath(dst), path):
                            sym_target = pathlib.Path(dst)
                            if fileish.exists():
                                fileish.unlink()
                            fileish.symlink_to(sym_target)
                        else:
                            raise Bad7zFile('Symlink point out of target directory.')
                else:
                    with fileish.open(mode='wb') as obfp:
                        crc32 = self.decompress(fp, f.folder, obfp, f.uncompressed, f.compressed, src_end, q)
                        obfp.seek(0)
                        if f.crc32 is not None and crc32 != f.crc32:
                            raise CrcError(crc32, f.crc32, f.filename)
            elif not isinstance(fileish, MemIO):
                fileish.touch()
            else:
                with fileish.open() as ofp:
                    pass
        if q is not None:
            q.put(('e', str(f.filename), str(f.uncompressed)))
    if not skip_notarget:
        self._check(fp, just_check, src_end)