from __future__ import annotations
import gzip
import io
import typing as ty
from bz2 import BZ2File
from os.path import splitext
from ._compression import HAVE_INDEXED_GZIP, IndexedGzipFile, pyzstd
def _get_opener_argnames(self, fileish: str) -> OpenerDef:
    _, ext = splitext(fileish)
    if self.compress_ext_icase:
        ext = ext.lower()
        for key in self.compress_ext_map:
            if key is None:
                continue
            if key.lower() == ext:
                return self.compress_ext_map[key]
    elif ext in self.compress_ext_map:
        return self.compress_ext_map[ext]
    return self.compress_ext_map[None]