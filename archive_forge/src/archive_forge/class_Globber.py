from __future__ import unicode_literals
import typing
import re
from collections import namedtuple
from . import wildcard
from ._repr import make_repr
from .lrucache import LRUCache
from .path import iteratepath
class Globber(object):
    """A generator of glob results."""

    def __init__(self, fs, pattern, path='/', namespaces=None, case_sensitive=True, exclude_dirs=None):
        """Create a new Globber instance.

        Arguments:
            fs (~fs.base.FS): A filesystem object
            pattern (str): A glob pattern, e.g. ``"**/*.py"``
            path (str): A path to a directory in the filesystem.
            namespaces (list): A list of additional info namespaces.
            case_sensitive (bool): If ``True``, the path matching will be
                case *sensitive* i.e. ``"FOO.py"`` and ``"foo.py"`` will be
                different, otherwise path matching will be case *insensitive*.
            exclude_dirs (list): A list of patterns to exclude when searching,
                e.g. ``["*.git"]``.

        """
        self.fs = fs
        self.pattern = pattern
        self.path = path
        self.namespaces = namespaces
        self.case_sensitive = case_sensitive
        self.exclude_dirs = exclude_dirs

    def __repr__(self):
        return make_repr(self.__class__.__name__, self.fs, self.pattern, path=(self.path, '/'), namespaces=(self.namespaces, None), case_sensitive=(self.case_sensitive, True), exclude_dirs=(self.exclude_dirs, None))

    def _make_iter(self, search='breadth', namespaces=None):
        try:
            levels, recursive, re_pattern = _PATTERN_CACHE[self.pattern, self.case_sensitive]
        except KeyError:
            levels, recursive, re_pattern = _translate_glob(self.pattern, case_sensitive=self.case_sensitive)
        for path, info in self.fs.walk.info(path=self.path, namespaces=namespaces or self.namespaces, max_depth=None if recursive else levels, search=search, exclude_dirs=self.exclude_dirs):
            if info.is_dir:
                path += '/'
            if re_pattern.match(path):
                yield GlobMatch(path, info)

    def __iter__(self):
        """Get an iterator of :class:`fs.glob.GlobMatch` objects."""
        return self._make_iter()

    def count(self):
        """Count files / directories / data in matched paths.

        Example:
            >>> my_fs.glob('**/*.py').count()
            Counts(files=2, directories=0, data=55)

        Returns:
            `~Counts`: A named tuple containing results.

        """
        directories = 0
        files = 0
        data = 0
        for _path, info in self._make_iter(namespaces=['details']):
            if info.is_dir:
                directories += 1
            else:
                files += 1
            data += info.size
        return Counts(directories=directories, files=files, data=data)

    def count_lines(self):
        """Count the lines in the matched files.

        Returns:
            `~LineCounts`: A named tuple containing line counts.

        Example:
            >>> my_fs.glob('**/*.py').count_lines()
            LineCounts(lines=4, non_blank=3)

        """
        lines = 0
        non_blank = 0
        for path, info in self._make_iter():
            if info.is_file:
                for line in self.fs.open(path, 'rb'):
                    lines += 1
                    if line.rstrip():
                        non_blank += 1
        return LineCounts(lines=lines, non_blank=non_blank)

    def remove(self):
        """Remove all matched paths.

        Returns:
            int: Number of file and directories removed.

        Example:
            >>> my_fs.glob('**/*.pyc').remove()
            2

        """
        removes = 0
        for path, info in self._make_iter(search='depth'):
            if info.is_dir:
                self.fs.removetree(path)
            else:
                self.fs.remove(path)
            removes += 1
        return removes