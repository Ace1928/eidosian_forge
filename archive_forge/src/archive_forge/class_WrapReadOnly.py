from __future__ import print_function, unicode_literals
import typing
from .errors import ResourceNotFound, ResourceReadOnly
from .info import Info
from .mode import check_writable
from .path import abspath, normpath, split
from .wrapfs import WrapFS
class WrapReadOnly(WrapFS[_F], typing.Generic[_F]):
    """Makes a Filesystem read-only.

    Any call that would would write data or modify the filesystem in any way
    will raise a `~fs.errors.ResourceReadOnly` exception.

    """
    wrap_name = 'read-only'

    def appendbytes(self, path, data):
        self.check()
        raise ResourceReadOnly(path)

    def appendtext(self, path, text, encoding='utf-8', errors=None, newline=''):
        self.check()
        raise ResourceReadOnly(path)

    def makedir(self, path, permissions=None, recreate=False):
        self.check()
        raise ResourceReadOnly(path)

    def move(self, src_path, dst_path, overwrite=False, preserve_time=False):
        self.check()
        raise ResourceReadOnly(dst_path)

    def openbin(self, path, mode='r', buffering=-1, **options):
        self.check()
        if check_writable(mode):
            raise ResourceReadOnly(path)
        return self._wrap_fs.openbin(path, mode=mode, buffering=-1, **options)

    def remove(self, path):
        self.check()
        raise ResourceReadOnly(path)

    def removedir(self, path):
        self.check()
        raise ResourceReadOnly(path)

    def removetree(self, path):
        self.check()
        raise ResourceReadOnly(path)

    def setinfo(self, path, info):
        self.check()
        raise ResourceReadOnly(path)

    def writetext(self, path, contents, encoding='utf-8', errors=None, newline=''):
        self.check()
        raise ResourceReadOnly(path)

    def settimes(self, path, accessed=None, modified=None):
        self.check()
        raise ResourceReadOnly(path)

    def copy(self, src_path, dst_path, overwrite=False, preserve_time=False):
        self.check()
        raise ResourceReadOnly(dst_path)

    def create(self, path, wipe=False):
        self.check()
        raise ResourceReadOnly(path)

    def makedirs(self, path, permissions=None, recreate=False):
        self.check()
        raise ResourceReadOnly(path)

    def open(self, path, mode='r', buffering=-1, encoding=None, errors=None, newline='', line_buffering=False, **options):
        self.check()
        if check_writable(mode):
            raise ResourceReadOnly(path)
        return self._wrap_fs.open(path, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, line_buffering=line_buffering, **options)

    def writebytes(self, path, contents):
        self.check()
        raise ResourceReadOnly(path)

    def upload(self, path, file, chunk_size=None, **options):
        self.check()
        raise ResourceReadOnly(path)

    def writefile(self, path, file, encoding=None, errors=None, newline=''):
        self.check()
        raise ResourceReadOnly(path)

    def touch(self, path):
        self.check()
        raise ResourceReadOnly(path)

    def getmeta(self, namespace='standard'):
        self.check()
        meta = dict(self.delegate_fs().getmeta(namespace=namespace))
        meta.update(read_only=True, supports_rename=False)
        return meta