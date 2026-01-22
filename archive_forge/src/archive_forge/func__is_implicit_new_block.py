import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _is_implicit_new_block(self, state):
    inst = state.get_inst()
    if inst.offset in self._bytecode.labels:
        return True
    elif inst.opname in NEW_BLOCKERS:
        return True
    else:
        return False