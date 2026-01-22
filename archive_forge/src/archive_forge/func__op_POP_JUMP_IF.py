import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _op_POP_JUMP_IF(self, state, inst):
    pred = state.pop()
    state.append(inst, pred=pred)
    target_inst = inst.get_jump_target()
    next_inst = inst.next
    state.fork(pc=next_inst)
    if target_inst != next_inst:
        state.fork(pc=target_inst)