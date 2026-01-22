import os
import posixpath
import sys
import urllib.parse
import warnings
from os.path import join as pjoin
import pyarrow as pa
from pyarrow.util import doc, _stringify_path, _is_path_like, _DEPR_MSG
class DaskFileSystem(FileSystem):
    """
    Wraps s3fs Dask filesystem implementation like s3fs, gcsfs, etc.
    """

    def __init__(self, fs):
        warnings.warn('The pyarrow.filesystem.DaskFileSystem/S3FSWrapper are deprecated as of pyarrow 3.0.0, and will be removed in a future version.', FutureWarning, stacklevel=2)
        self.fs = fs

    @doc(FileSystem.isdir)
    def isdir(self, path):
        raise NotImplementedError('Unsupported file system API')

    @doc(FileSystem.isfile)
    def isfile(self, path):
        raise NotImplementedError('Unsupported file system API')

    @doc(FileSystem._isfilestore)
    def _isfilestore(self):
        """
        Object Stores like S3 and GCSFS are based on key lookups, not true
        file-paths.
        """
        return False

    @doc(FileSystem.delete)
    def delete(self, path, recursive=False):
        path = _stringify_path(path)
        return self.fs.rm(path, recursive=recursive)

    @doc(FileSystem.exists)
    def exists(self, path):
        path = _stringify_path(path)
        return self.fs.exists(path)

    @doc(FileSystem.mkdir)
    def mkdir(self, path, create_parents=True):
        path = _stringify_path(path)
        if create_parents:
            return self.fs.mkdirs(path)
        else:
            return self.fs.mkdir(path)

    @doc(FileSystem.open)
    def open(self, path, mode='rb'):
        """
        Open file for reading or writing.
        """
        path = _stringify_path(path)
        return self.fs.open(path, mode=mode)

    def ls(self, path, detail=False):
        path = _stringify_path(path)
        return self.fs.ls(path, detail=detail)

    def walk(self, path):
        """
        Directory tree generator, like os.walk.
        """
        path = _stringify_path(path)
        return self.fs.walk(path)