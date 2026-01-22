import contextlib
import errno
import io
import os
import shutil
import cachetools
import fasteners
from oslo_serialization import jsonutils
from oslo_utils import fileutils
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.utils import misc
class DirBackend(path_based.PathBasedBackend):
    """A directory and file based backend.

    This backend does *not* provide true transactional semantics. It does
    guarantee that there will be no interprocess race conditions when
    writing and reading by using a consistent hierarchy of file based locks.

    Example configuration::

        conf = {
            "path": "/tmp/taskflow",  # save data to this root directory
            "max_cache_size": 1024,  # keep up-to 1024 entries in memory
        }
    """
    DEFAULT_FILE_ENCODING = 'utf-8'
    '\n    Default encoding used when decoding or encoding files into or from\n    text/unicode into binary or binary into text/unicode.\n    '

    def __init__(self, conf):
        super(DirBackend, self).__init__(conf)
        max_cache_size = self._conf.get('max_cache_size')
        if max_cache_size is not None:
            max_cache_size = int(max_cache_size)
            if max_cache_size < 1:
                raise ValueError('Maximum cache size must be greater than or equal to one')
            self.file_cache = cachetools.LRUCache(max_cache_size)
        else:
            self.file_cache = {}
        self.encoding = self._conf.get('encoding', self.DEFAULT_FILE_ENCODING)
        if not self._path:
            raise ValueError('Empty path is disallowed')
        self._path = os.path.abspath(self._path)
        self.lock = fasteners.ReaderWriterLock()

    def get_connection(self):
        return Connection(self)

    def close(self):
        pass