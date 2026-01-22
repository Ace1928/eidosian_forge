from collections import deque
from . import errors, revision
class _MergeSortNode:
    """Information about a specific node in the merge graph."""
    __slots__ = ('key', 'merge_depth', 'revno', 'end_of_merge')

    def __init__(self, key, merge_depth, revno, end_of_merge):
        self.key = key
        self.merge_depth = merge_depth
        self.revno = revno
        self.end_of_merge = end_of_merge