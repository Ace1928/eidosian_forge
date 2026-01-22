import abc
from collections.abc import Mapping
from typing import TypeVar, Generic
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
class RegionVisitor(abc.ABC, Generic[Tdata]):
    """A non-mutating pass on a SCFG.

    When each block is visited, their parent must have be visited.
    The abstract ``visit_*`` methods will receive and will return any custom
    data of type Tdata.
    """
    direction = 'forward'
    'The direction in which the graph is processed. Default is set to\n    "forward". Set to "backward" for reverse dataflow direction.\n    '

    @abc.abstractmethod
    def visit_block(self, block: BasicBlock, data: Tdata) -> Tdata:
        """This is called when a BasicBlock is visited."""
        pass

    @abc.abstractmethod
    def visit_loop(self, region: RegionBlock, data: Tdata) -> Tdata:
        """This is called when a loop region is visited.

        When overriding this method, remember to handle the merging path of
        ``data`` for the backedge back to the head of the loop.
        """
        pass

    @abc.abstractmethod
    def visit_switch(self, region: RegionBlock, data: Tdata) -> Tdata:
        """This is called when a switch region is visited.

        When overriding this method, remember to handle the merging path of
        ``data`` for all the branches in the switch when joining into the tail.
        """
        pass

    def visit_linear(self, region: RegionBlock, data: Tdata) -> Tdata:
        """This is called when a linear region is visited."""
        return self.visit_graph(region.subregion, data)

    def visit_graph(self, scfg: SCFG, data: Tdata) -> Tdata:
        """Process a SCFG in topological order."""
        toposorted = self._toposort_graph(scfg)
        label: str
        for lvl in toposorted:
            for label in lvl:
                data = self.visit(scfg[label], data)
        return data

    def _toposort_graph(self, scfg: SCFG):
        toposorted = toposort_graph(scfg.graph)
        if self.direction == 'forward':
            return toposorted
        elif self.direction == 'backward':
            return reversed(toposorted)
        else:
            assert False, f'invalid direction {self.direction!r}'

    def visit(self, block: BasicBlock, data: Tdata) -> Tdata:
        """A generic visit method that will dispatch to the correct"""
        if isinstance(block, RegionBlock):
            if block.kind == 'loop':
                fn = self.visit_loop
            elif block.kind == 'switch':
                fn = self.visit_switch
            else:
                raise NotImplementedError('unreachable')
            data = fn(block, data)
        else:
            data = self.visit_block(block, data)
        return data