from __future__ import unicode_literals
import typing
from collections import defaultdict, deque, namedtuple
from ._repr import make_repr
from .errors import FSError
from .path import abspath, combine, normpath
def _walk_depth(self, fs, path, namespaces=None):
    """Walk files using a *depth first* search."""
    _combine = combine
    _scan = self._scan
    _calculate_depth = self._calculate_depth
    _check_open_dir = self._check_open_dir
    _check_scan_dir = self._check_scan_dir
    _check_file = self.check_file
    depth = _calculate_depth(path)
    stack = [(path, _scan(fs, path, namespaces=namespaces), None)]
    push = stack.append
    while stack:
        dir_path, iter_files, parent = stack[-1]
        info = next(iter_files, None)
        if info is None:
            if parent is not None:
                yield parent
            yield (dir_path, None)
            del stack[-1]
        elif info.is_dir:
            _depth = _calculate_depth(dir_path) - depth + 1
            if _check_open_dir(fs, dir_path, info):
                if _check_scan_dir(fs, dir_path, info, _depth):
                    _path = _combine(dir_path, info.name)
                    push((_path, _scan(fs, _path, namespaces=namespaces), (dir_path, info)))
                else:
                    yield (dir_path, info)
        elif _check_file(fs, info):
            yield (dir_path, info)