import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _guard_with_as(self, state):
    """Checks if the next instruction after a SETUP_WITH is something other
        than a POP_TOP, if it is something else it'll be some sort of store
        which is not supported (this corresponds to `with CTXMGR as VAR(S)`)."""
    current_inst = state.get_inst()
    if current_inst.opname in {'SETUP_WITH', 'BEFORE_WITH'}:
        next_op = self._bytecode[current_inst.next].opname
        if next_op != 'POP_TOP':
            msg = "The 'with (context manager) as (variable):' construct is not supported."
            raise UnsupportedError(msg)