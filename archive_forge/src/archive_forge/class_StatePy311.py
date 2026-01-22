import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
class StatePy311(_State):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kw_names = None

    def pop_kw_names(self):
        out = self._kw_names
        self._kw_names = None
        return out

    def set_kw_names(self, val):
        assert self._kw_names is None
        self._kw_names = val

    def is_in_exception(self):
        bc = self._bytecode
        return bc.find_exception_entry(self._pc) is not None

    def get_exception(self):
        bc = self._bytecode
        return bc.find_exception_entry(self._pc)

    def in_with(self):
        for ent in self._blockstack_initial:
            if ent['kind'] == BlockKind('WITH'):
                return True

    def make_null(self):
        return self.make_temp(prefix='null$')