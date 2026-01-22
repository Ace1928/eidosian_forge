from __future__ import annotations
import gzip
import io
import typing as ty
from bz2 import BZ2File
from os.path import splitext
from ._compression import HAVE_INDEXED_GZIP, IndexedGzipFile, pyzstd
def _gzip_open(filename: str, mode: Mode='rb', compresslevel: int=9, mtime: int=0, keep_open: bool=False) -> gzip.GzipFile:
    if not HAVE_INDEXED_GZIP or mode != 'rb':
        gzip_file = DeterministicGzipFile(filename, mode, compresslevel, mtime=mtime)
    else:
        gzip_file = IndexedGzipFile(filename, drop_handles=not keep_open)
    return gzip_file