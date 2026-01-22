import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _run_handle_exception(self, runner, state):
    if state.has_active_try() and state.get_inst().opname not in _NO_RAISE_OPS:
        state.fork(pc=state.get_inst().next)
        tryblk = state.get_top_block('TRY')
        state.pop_block_and_above(tryblk)
        nstack = state.stack_depth
        kwargs = {}
        if nstack > tryblk['entry_stack']:
            kwargs['npop'] = nstack - tryblk['entry_stack']
        handler = tryblk['handler']
        kwargs['npush'] = {BlockKind('EXCEPT'): _EXCEPT_STACK_OFFSET, BlockKind('FINALLY'): _FINALLY_POP}[handler['kind']]
        kwargs['extra_block'] = handler
        state.fork(pc=tryblk['end'], **kwargs)
        return True
    else:
        state.advance_pc()