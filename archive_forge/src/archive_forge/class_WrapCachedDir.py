from __future__ import print_function, unicode_literals
import typing
from .errors import ResourceNotFound, ResourceReadOnly
from .info import Info
from .mode import check_writable
from .path import abspath, normpath, split
from .wrapfs import WrapFS
class WrapCachedDir(WrapFS[_F], typing.Generic[_F]):
    """Caches filesystem directory information.

    This filesystem caches directory information retrieved from a
    scandir call. This *may* speed up code that calls `~FS.isdir`,
    `~FS.isfile`, or `~FS.gettype` too frequently.

    Note:
        Using this wrap will prevent changes to directory information
        being visible to the filesystem object. Consequently it is best
        used only in a fairly limited scope where you don't expected
        anything on the filesystem to change.

    """
    wrap_name = 'cached-dir'

    def __init__(self, wrap_fs):
        super(WrapCachedDir, self).__init__(wrap_fs)
        self._cache = {}

    def scandir(self, path, namespaces=None, page=None):
        _path = abspath(normpath(path))
        cache_key = (_path, frozenset(namespaces or ()))
        if cache_key not in self._cache:
            _scan_result = self._wrap_fs.scandir(path, namespaces=namespaces, page=page)
            _dir = {info.name: info for info in _scan_result}
            self._cache[cache_key] = _dir
        gen_scandir = iter(self._cache[cache_key].values())
        return gen_scandir

    def getinfo(self, path, namespaces=None):
        _path = abspath(normpath(path))
        if _path == '/':
            return Info({'basic': {'name': '', 'is_dir': True}})
        dir_path, resource_name = split(_path)
        cache_key = (dir_path, frozenset(namespaces or ()))
        if cache_key not in self._cache:
            self.scandir(dir_path, namespaces=namespaces)
        _dir = self._cache[cache_key]
        try:
            info = _dir[resource_name]
        except KeyError:
            raise ResourceNotFound(path)
        return info

    def isdir(self, path):
        try:
            return self.getinfo(path).is_dir
        except ResourceNotFound:
            return False

    def isfile(self, path):
        try:
            return not self.getinfo(path).is_dir
        except ResourceNotFound:
            return False