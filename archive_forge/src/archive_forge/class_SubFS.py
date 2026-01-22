from __future__ import print_function, unicode_literals
import typing
import six
from .path import abspath, join, normpath, relpath
from .wrapfs import WrapFS
@six.python_2_unicode_compatible
class SubFS(WrapFS[_F], typing.Generic[_F]):
    """A sub-directory on a parent filesystem.

    A SubFS is a filesystem object that maps to a sub-directory of
    another filesystem. This is the object that is returned by
    `~fs.base.FS.opendir`.

    """

    def __init__(self, parent_fs, path):
        super(SubFS, self).__init__(parent_fs)
        self._sub_dir = abspath(normpath(path))

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self._wrap_fs, self._sub_dir)

    def __str__(self):
        return '{parent}{dir}'.format(parent=self._wrap_fs, dir=self._sub_dir)

    def delegate_fs(self):
        return self._wrap_fs

    def delegate_path(self, path):
        _path = join(self._sub_dir, relpath(normpath(path)))
        return (self._wrap_fs, _path)