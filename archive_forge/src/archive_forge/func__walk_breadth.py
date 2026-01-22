from __future__ import unicode_literals
import typing
from collections import defaultdict, deque, namedtuple
from ._repr import make_repr
from .errors import FSError
from .path import abspath, combine, normpath
def _walk_breadth(self, fs, path, namespaces=None):
    """Walk files using a *breadth first* search."""
    queue = deque([path])
    push = queue.appendleft
    pop = queue.pop
    _combine = combine
    _scan = self._scan
    _calculate_depth = self._calculate_depth
    _check_open_dir = self._check_open_dir
    _check_scan_dir = self._check_scan_dir
    _check_file = self.check_file
    depth = _calculate_depth(path)
    while queue:
        dir_path = pop()
        for info in _scan(fs, dir_path, namespaces=namespaces):
            if info.is_dir:
                _depth = _calculate_depth(dir_path) - depth + 1
                if _check_open_dir(fs, dir_path, info):
                    yield (dir_path, info)
                    if _check_scan_dir(fs, dir_path, info, _depth):
                        push(_combine(dir_path, info.name))
            elif _check_file(fs, info):
                yield (dir_path, info)
        yield (dir_path, None)