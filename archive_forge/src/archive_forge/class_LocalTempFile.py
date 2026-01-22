from __future__ import annotations
import inspect
import logging
import os
import tempfile
import time
import weakref
from shutil import rmtree
from typing import TYPE_CHECKING, Any, Callable, ClassVar
from fsspec import AbstractFileSystem, filesystem
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.compression import compr
from fsspec.core import BaseCache, MMapCache
from fsspec.exceptions import BlocksizeMismatchError
from fsspec.implementations.cache_mapper import create_cache_mapper
from fsspec.implementations.cache_metadata import CacheMetadata
from fsspec.spec import AbstractBufferedFile
from fsspec.transaction import Transaction
from fsspec.utils import infer_compression
class LocalTempFile:
    """A temporary local file, which will be uploaded on commit"""

    def __init__(self, fs, path, fn, mode='wb', autocommit=True, seek=0, **kwargs):
        self.fn = fn
        self.fh = open(fn, mode)
        self.mode = mode
        if seek:
            self.fh.seek(seek)
        self.path = path
        self.size = None
        self.fs = fs
        self.closed = False
        self.autocommit = autocommit
        self.kwargs = kwargs

    def __reduce__(self):
        return (LocalTempFile, (self.fs, self.path, self.fn, 'r+b', self.autocommit, self.tell()))

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.size = self.fh.tell()
        if self.closed:
            return
        self.fh.close()
        self.closed = True
        if self.autocommit:
            self.commit()

    def discard(self):
        self.fh.close()
        os.remove(self.fn)

    def commit(self):
        self.fs.put(self.fn, self.path, **self.kwargs)

    @property
    def name(self):
        return self.fn

    def __repr__(self) -> str:
        return f'LocalTempFile: {self.path}'

    def __getattr__(self, item):
        return getattr(self.fh, item)