from __future__ import print_function, unicode_literals
import sys
import typing
import six
import zipfile
from datetime import datetime
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_zip
from .enums import ResourceType, Seek
from .info import Info
from .iotools import RawWrapper
from .memoryfs import MemoryFS
from .opener import open_fs
from .path import dirname, forcedir, normpath, relpath
from .permissions import Permissions
from .time import datetime_to_epoch
from .wrapfs import WrapFS
@property
def _directory(self):
    """`MemoryFS`: a filesystem with the same folder hierarchy as the zip."""
    self.check()
    with self._lock:
        if self._directory_fs is None:
            self._directory_fs = _fs = MemoryFS()
            for zip_name in self._zip.namelist():
                resource_name = zip_name
                if six.PY2:
                    resource_name = resource_name.decode(self.encoding, 'replace')
                if resource_name.endswith('/'):
                    _fs.makedirs(resource_name, recreate=True)
                else:
                    _fs.makedirs(dirname(resource_name), recreate=True)
                    _fs.create(resource_name)
        return self._directory_fs