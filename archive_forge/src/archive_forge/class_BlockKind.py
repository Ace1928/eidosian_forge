import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
@total_ordering
class BlockKind(object):
    """Kinds of block to make related code safer than just `str`.
    """
    _members = frozenset({'LOOP', 'TRY', 'EXCEPT', 'FINALLY', 'WITH', 'WITH_FINALLY'})

    def __init__(self, value):
        assert value in self._members
        self._value = value

    def __hash__(self):
        return hash((type(self), self._value))

    def __lt__(self, other):
        if isinstance(other, BlockKind):
            return self._value < other._value
        else:
            raise TypeError('cannot compare to {!r}'.format(type(other)))

    def __eq__(self, other):
        if isinstance(other, BlockKind):
            return self._value == other._value
        else:
            raise TypeError('cannot compare to {!r}'.format(type(other)))

    def __repr__(self):
        return 'BlockKind({})'.format(self._value)