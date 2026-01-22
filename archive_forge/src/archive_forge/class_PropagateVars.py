import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
class PropagateVars(RegionVisitor[_pvData]):
    """
    Depends on PropagateStack
    """

    def __init__(self, _debug: bool=False):
        super().__init__()
        self._debug = _debug

    def debug_print(self, *args, **kwargs):
        if self._debug:
            print(*args, **kwargs)

    def _apply(self, block: BasicBlock, data: _pvData) -> _pvData:
        assert isinstance(block, BasicBlock)
        if isinstance(block, DDGProtocol):
            if isinstance(block, DDGBlock):
                for k in data:
                    if k not in block.in_vars:
                        op = Op(opname='var.incoming', bc_inst=None)
                        vs = op.add_output(k)
                        block.in_vars[k] = vs
                        if k.startswith('tos.'):
                            if k in block.exported_stackvars:
                                block.out_vars[k] = block.exported_stackvars[k]
                        elif k not in block.out_vars:
                            block.out_vars[k] = vs
            else:
                block.incoming_states.update(data)
                block.outgoing_states.update(data)
            data = set(block.outgoing_states)
            return data
        else:
            return data

    def visit_linear(self, region: RegionBlock, data: _pvData) -> _pvData:
        region.incoming_states.update(data)
        data = self.visit_graph(region.subregion, data)
        region.outgoing_states.update(data)
        return set(region.outgoing_states)

    def visit_block(self, block: BasicBlock, data: _pvData) -> _pvData:
        return self._apply(block, data)

    def visit_loop(self, region: RegionBlock, data: _pvData) -> _pvData:
        self.debug_print('---LOOP_ENTER', region.name, data)
        data = self.visit_linear(region, data)
        self.debug_print('---LOOP_END=', region.name, 'vars', data)
        return data

    def visit_switch(self, region: RegionBlock, data: _pvData) -> _pvData:
        self.debug_print('---SWITCH_ENTER', region.name)
        region.incoming_states.update(data)
        header = region.header
        data_at_head = self.visit_linear(region.subregion[header], data)
        data_for_branches = []
        for blk in region.subregion.graph.values():
            if blk.kind == 'branch':
                data_for_branches.append(self.visit_linear(blk, data_at_head))
        data_after_branches = reduce(operator.or_, data_for_branches)
        exiting = region.exiting
        data_at_tail = self.visit_linear(region.subregion[exiting], data_after_branches)
        self.debug_print('data_at_head', data_at_head)
        self.debug_print('data_for_branches', data_for_branches)
        self.debug_print('data_after_branches', data_after_branches)
        self.debug_print('data_at_tail', data_at_tail)
        self.debug_print('---SWITCH_END=', region.name, 'vars', data_at_tail)
        region.outgoing_states.update(data_at_tail)
        return set(region.outgoing_states)

    def make_data(self) -> _pvData:
        return set()