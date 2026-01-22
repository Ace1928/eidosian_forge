import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class _CoalescedOffset:
    """A data container for keeping track of coalesced offsets."""
    __slots__ = ['start', 'length', 'ranges']

    def __init__(self, start, length, ranges):
        self.start = start
        self.length = length
        self.ranges = ranges

    def __lt__(self, other):
        return (self.start, self.length, self.ranges) < (other.start, other.length, other.ranges)

    def __eq__(self, other):
        return (self.start, self.length, self.ranges) == (other.start, other.length, other.ranges)

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__, self.start, self.length, self.ranges)