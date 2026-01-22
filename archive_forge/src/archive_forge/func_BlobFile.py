from __future__ import annotations
import concurrent.futures
from typing import (
import urllib3
from blobfile._common import DirEntry, Stat, RemoteOrLocalPath
from blobfile._context import (
def BlobFile(path: RemoteOrLocalPath, mode: Literal['r', 'rb', 'w', 'wb', 'a', 'ab']='r', streaming: Optional[bool]=None, buffer_size: Optional[int]=None, cache_dir: Optional[str]=None, file_size: Optional[int]=None, version: Optional[str]=None):
    """
    Open a local or remote file for reading or writing

    Args:
        path local or remote path
        mode: one of "r", "rb", "w", "wb", "a", "ab" indicating the mode to open the file in
        streaming: the default for `streaming` is `True` when `mode` is in `"r", "rb"` and `False` when `mode` is in `"w", "wb", "a", "ab"`.
            * `streaming=True`:
                * Reading is done without downloading the entire remote file.
                * Writing is done to the remote file directly, but only in chunks of a few MB in size.  `flush()` will not cause an early write.
                * Appending is not implemented.
            * `streaming=False`:
                * Reading is done by downloading the remote file to a local file during the constructor.
                * Writing is done by uploading the file on `close()` or during destruction.
                * Appending is done by downloading the file during construction and uploading on `close()` or during destruction.
        buffer_size: number of bytes to buffer, this can potentially make reading more efficient.
        cache_dir: a directory in which to cache files for reading, only valid if `streaming=False` and `mode` is in `"r", "rb"`.   You are reponsible for cleaning up the cache directory.

    Returns:
        A file-like object
    """
    return default_context.BlobFile(path=path, mode=mode, streaming=streaming, buffer_size=buffer_size, cache_dir=cache_dir, file_size=file_size, version=version)