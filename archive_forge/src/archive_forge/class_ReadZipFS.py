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
@six.python_2_unicode_compatible
class ReadZipFS(FS):
    """A readable zip file."""
    _meta = {'case_insensitive': False, 'network': False, 'read_only': True, 'supports_rename': False, 'thread_safe': True, 'unicode_paths': True, 'virtual': False}

    @errors.CreateFailed.catch_all
    def __init__(self, file, encoding='utf-8'):
        super(ReadZipFS, self).__init__()
        self._file = file
        self.encoding = encoding
        self._zip = zipfile.ZipFile(file, 'r')
        self._directory_fs = None

    def __repr__(self):
        return 'ReadZipFS({!r})'.format(self._file)

    def __str__(self):
        return "<zipfs '{}'>".format(self._file)

    def _path_to_zip_name(self, path):
        """Convert a path to a zip file name."""
        path = relpath(normpath(path))
        if self._directory.isdir(path):
            path = forcedir(path)
        if six.PY2:
            return path.encode(self.encoding)
        return path

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

    def getinfo(self, path, namespaces=None):
        _path = self.validatepath(path)
        namespaces = namespaces or ()
        raw_info = {}
        if _path == '/':
            raw_info['basic'] = {'name': '', 'is_dir': True}
            if 'details' in namespaces:
                raw_info['details'] = {'type': int(ResourceType.directory)}
        else:
            basic_info = self._directory.getinfo(_path)
            raw_info['basic'] = {'name': basic_info.name, 'is_dir': basic_info.is_dir}
            if not {'details', 'access', 'zip'}.isdisjoint(namespaces):
                zip_name = self._path_to_zip_name(path)
                try:
                    zip_info = self._zip.getinfo(zip_name)
                except KeyError:
                    pass
                else:
                    if 'details' in namespaces:
                        raw_info['details'] = {'size': zip_info.file_size, 'type': int(ResourceType.directory if basic_info.is_dir else ResourceType.file), 'modified': datetime_to_epoch(datetime(*zip_info.date_time))}
                    if 'zip' in namespaces:
                        raw_info['zip'] = {k: getattr(zip_info, k) for k in zip_info.__slots__ if not k.startswith('_')}
                    if 'access' in namespaces:
                        if zip_info.external_attr and zip_info.create_system == 3:
                            raw_info['access'] = {'permissions': Permissions(mode=zip_info.external_attr >> 16 & 4095).dump()}
        return Info(raw_info)

    def setinfo(self, path, info):
        self.check()
        raise errors.ResourceReadOnly(path)

    def listdir(self, path):
        self.check()
        return self._directory.listdir(path)

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        raise errors.ResourceReadOnly(path)

    def openbin(self, path, mode='r', buffering=-1, **kwargs):
        self.check()
        if 'w' in mode or '+' in mode or 'a' in mode:
            raise errors.ResourceReadOnly(path)
        if not self._directory.exists(path):
            raise errors.ResourceNotFound(path)
        elif self._directory.isdir(path):
            raise errors.FileExpected(path)
        zip_name = self._path_to_zip_name(path)
        return _ZipExtFile(self, zip_name)

    def remove(self, path):
        self.check()
        raise errors.ResourceReadOnly(path)

    def removedir(self, path):
        self.check()
        raise errors.ResourceReadOnly(path)

    def close(self):
        super(ReadZipFS, self).close()
        if hasattr(self, '_zip'):
            self._zip.close()

    def readbytes(self, path):
        self.check()
        if not self._directory.isfile(path):
            raise errors.ResourceNotFound(path)
        zip_name = self._path_to_zip_name(path)
        zip_bytes = self._zip.read(zip_name)
        return zip_bytes

    def geturl(self, path, purpose='download'):
        if purpose == 'fs' and isinstance(self._file, six.string_types):
            quoted_file = url_quote(self._file)
            quoted_path = url_quote(path)
            return 'zip://{}!/{}'.format(quoted_file, quoted_path)
        else:
            raise errors.NoURL(path, purpose)