from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _check_delta_unique_old_paths(delta):
    """Decorate a delta and check that the old paths in it are unique.

    :return: A generator over delta.
    """
    paths = set()
    for item in delta:
        length = len(paths) + 1
        path = item[0]
        if path is not None:
            paths.add(path)
            if len(paths) != length:
                raise errors.InconsistentDelta(path, item[2], 'repeated path')
        yield item