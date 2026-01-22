import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
class GzipFileSystem(BaseCompressedFileFileSystem):
    """Read contents of GZIP file as a filesystem with one file inside."""
    protocol = 'gzip'
    compression = 'gzip'
    extension = '.gz'