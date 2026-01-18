from contextlib import contextmanager
from ctypes import (
import libarchive
import libarchive.ffi as ffi
from fsspec import open_files
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.utils import DEFAULT_BLOCK_SIZE

        Parameters
        ----------
        fo: str or file-like
            Contains ZIP, and must exist. If a str, will fetch file using
            :meth:`~fsspec.open_files`, which must return one file exactly.
        mode: str
            Currently, only 'r' accepted
        target_protocol: str (optional)
            If ``fo`` is a string, this value can be used to override the
            FS protocol inferred from a URL
        target_options: dict (optional)
            Kwargs passed when instantiating the target FS, if ``fo`` is
            a string.
        