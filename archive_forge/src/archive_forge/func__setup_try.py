import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _setup_try(self, kind, state, next, end):
    handler_block = state.make_block(kind=kind, end=None, reset_stack=False)
    state.fork(pc=next, extra_block=state.make_block(kind='TRY', end=end, reset_stack=False, handler=handler_block))