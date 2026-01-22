from __future__ import unicode_literals
import typing
import re
from collections import namedtuple
from . import wildcard
from ._repr import make_repr
from .lrucache import LRUCache
from .path import iteratepath
class BoundGlobber(object):
    """A `~fs.glob.Globber` object bound to a filesystem.

    An instance of this object is available on every Filesystem object
    as the `~fs.base.FS.glob` property.

    """
    __slots__ = ['fs']

    def __init__(self, fs):
        """Create a new bound Globber.

        Arguments:
            fs (FS): A filesystem object to bind to.

        """
        self.fs = fs

    def __repr__(self):
        return make_repr(self.__class__.__name__, self.fs)

    def __call__(self, pattern, path='/', namespaces=None, case_sensitive=True, exclude_dirs=None):
        """Match resources on the bound filesystem againsts a glob pattern.

        Arguments:
            pattern (str): A glob pattern, e.g. ``"**/*.py"``
            namespaces (list): A list of additional info namespaces.
            case_sensitive (bool): If ``True``, the path matching will be
                case *sensitive* i.e. ``"FOO.py"`` and ``"foo.py"`` will
                be different, otherwise path matching will be case **insensitive**.
            exclude_dirs (list): A list of patterns to exclude when searching,
                e.g. ``["*.git"]``.

        Returns:
            `Globber`: An object that may be queried for the glob matches.

        """
        return Globber(self.fs, pattern, path, namespaces=namespaces, case_sensitive=case_sensitive, exclude_dirs=exclude_dirs)