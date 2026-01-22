import binascii
import os
import struct
from .dirstate import DirState, DirstateCorrupt
def _bisect_path_right(paths, path):
    """Return the index where to insert path into paths.

    This uses a path-wise comparison so we get::
        a
        a-b
        a=b
        a/b
    Rather than::
        a
        a-b
        a/b
        a=b
    :param paths: A list of paths to search through
    :param path: A single path to insert
    :return: An offset where 'path' can be inserted.
    :seealso: bisect.bisect_right
    """
    hi = len(paths)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        cur = paths[mid]
        if _lt_path_by_dirblock(path, cur):
            hi = mid
        else:
            lo = mid + 1
    return lo