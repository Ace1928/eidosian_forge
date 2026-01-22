import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def _end_try_blocks(self):
    """Closes all try blocks by inserting the required marker at the
        exception handler

        This is only needed for py3.11 because of the changes in exception
        handling. This merely maps the new py3.11 semantics back to the old way.

        What the code does:

        - For each block, compute the difference of blockstack to its incoming
          blocks' blockstack.
        - If the incoming blockstack has an extra TRY, the current block must
          be the EXCEPT block and we need to insert a marker.

        See also: _insert_try_block_end
        """
    assert PYVERSION in ((3, 11), (3, 12))
    graph = self.cfa.graph
    for offset, block in self.blocks.items():
        cur_bs = self.dfa.infos[offset].blockstack
        for inc, _ in graph.predecessors(offset):
            inc_bs = self.dfa.infos[inc].blockstack
            for i, (x, y) in enumerate(zip(cur_bs, inc_bs)):
                if x != y:
                    break
            else:
                i = min(len(cur_bs), len(inc_bs))

            def do_change(remain):
                while remain:
                    ent = remain.pop()
                    if ent['kind'] == BlockKind('TRY'):
                        self.current_block = block
                        oldbody = list(block.body)
                        block.body.clear()
                        self._insert_try_block_end()
                        block.body.extend(oldbody)
                        return True
            if do_change(list(inc_bs[i:])):
                break