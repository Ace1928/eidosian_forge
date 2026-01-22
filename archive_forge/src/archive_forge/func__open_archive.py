from contextlib import contextmanager
from ctypes import (
import libarchive
import libarchive.ffi as ffi
from fsspec import open_files
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.utils import DEFAULT_BLOCK_SIZE
@contextmanager
def _open_archive(self):
    self.fo.seek(0)
    with custom_reader(self.fo, block_size=self.block_size) as arc:
        yield arc