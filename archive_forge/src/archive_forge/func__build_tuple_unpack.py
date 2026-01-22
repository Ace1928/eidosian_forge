import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _build_tuple_unpack(self, state, inst):
    tuples = list(reversed([state.pop() for _ in range(inst.arg)]))
    temps = [state.make_temp() for _ in range(len(tuples) - 1)]
    is_assign = len(tuples) == 1
    if is_assign:
        temps = [state.make_temp()]
    state.append(inst, tuples=tuples, temps=temps, is_assign=is_assign)
    state.push(temps[-1])