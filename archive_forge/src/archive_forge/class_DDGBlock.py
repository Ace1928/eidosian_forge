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
@dataclass(frozen=True)
class DDGBlock(BasicBlock):
    in_effect: ValueState | None = None
    out_effect: ValueState | None = None
    in_stackvars: list[ValueState] = field(default_factory=list)
    out_stackvars: list[ValueState] = field(default_factory=list)
    in_vars: MutableSortedMap[str, ValueState] = field(default_factory=MutableSortedMap)
    out_vars: MutableSortedMap[str, ValueState] = field(default_factory=MutableSortedMap)
    exported_stackvars: MutableSortedMap[str, ValueState] = field(default_factory=MutableSortedMap)

    def __post_init__(self):
        assert isinstance(self.in_vars, MutableSortedMap)
        assert isinstance(self.out_vars, MutableSortedMap)

    def _gather_reachable(self, vs: ValueState, reached: set[ValueState]) -> set[ValueState]:
        reached.add(vs)
        if vs.parent is not None:
            for ivs in vs.parent.inputs:
                if ivs not in reached:
                    self._gather_reachable(ivs, reached)
        return reached

    def render_graph(self, builder: 'GraphBuilder'):
        reached_vs: set[ValueState] = set()
        for vs in [*self.out_vars.values(), _just(self.out_effect)]:
            self._gather_reachable(vs, reached_vs)
        reached_vs.add(_just(self.in_effect))
        reached_vs.update(self.in_vars.values())
        reached_op = {vs.parent for vs in reached_vs if vs.parent is not None}
        for vs in reached_vs:
            self._render_vs(builder, vs)
        for op in reached_op:
            self._render_op(builder, op)
        ports = []
        outgoing_nodename = f'outgoing_{self.name}'
        for k, vs in self.out_vars.items():
            ports.append(k)
            builder.graph.add_edge(vs.short_identity(), outgoing_nodename, dst_port=k)
        outgoing_node = builder.node_maker.make_node(kind='ports', ports=ports, data=dict(body='outgoing'))
        builder.graph.add_node(outgoing_nodename, outgoing_node)
        ports = []
        incoming_nodename = f'incoming_{self.name}'
        for k, vs in self.in_vars.items():
            ports.append(k)
            builder.graph.add_edge(incoming_nodename, _just(vs.parent).short_identity(), src_port=k)
        incoming_node = builder.node_maker.make_node(kind='ports', ports=ports, data=dict(body='incoming'))
        builder.graph.add_node(incoming_nodename, incoming_node)
        jt_node = builder.node_maker.make_node(kind='meta', data=dict(body=f'jump-targets: {self._jump_targets}'))
        builder.graph.add_node(f'jt_{self.name}', jt_node)

    def _render_vs(self, builder: 'GraphBuilder', vs: ValueState):
        if vs.is_effect:
            node = builder.node_maker.make_node(kind='effect', data=dict(body=str(vs.name)))
            builder.graph.add_node(vs.short_identity(), node)
        else:
            node = builder.node_maker.make_node(kind='valuestate', data=dict(body=str(vs.name)))
            builder.graph.add_node(vs.short_identity(), node)

    def _render_op(self, builder, op: Op):
        op_anchor = op.short_identity()
        node = builder.node_maker.make_node(kind='op', data=dict(body=str(op.summary())))
        builder.graph.add_node(op_anchor, node)
        for edgename, vs in op._outputs.items():
            self._add_vs_edge(builder, op_anchor, vs, taillabel=f'{edgename}')
        for edgename, vs in op._inputs.items():
            self._add_vs_edge(builder, vs, op_anchor, headlabel=f'{edgename}')

    def _add_vs_edge(self, builder, src, dst, **attrs):
        is_effect = isinstance(src, ValueState) and src.is_effect or (isinstance(dst, ValueState) and dst.is_effect)
        if isinstance(src, ValueState):
            src = src.short_identity()
        if isinstance(dst, ValueState):
            dst = dst.short_identity()
        kwargs = attrs
        if is_effect:
            kwargs['kind'] = 'effect'
        builder.graph.add_edge(src, dst, **kwargs)

    @property
    def incoming_states(self) -> MutableSortedSet:
        return MutableSortedSet(self.in_vars)

    @property
    def outgoing_states(self) -> MutableSortedSet:
        return MutableSortedSet(self.out_vars)

    def get_toposorted_ops(self) -> list[Op]:
        """Get a topologically sorted list of ``Op`` according
        to the data-dependence.

        Operations stored later in the list may depend on earlier operations,
        but the reverse can never be true.
        """
        res: list[Op] = []
        avail: set[ValueState] = {*self.in_vars.values(), _just(self.in_effect)}
        pending: list[Op] = [vs.parent for vs in self.out_vars.values() if vs.parent is not None]
        assert self.out_effect is not None
        pending.append(_just(self.out_effect.parent))
        seen: set[Op] = set()
        while pending:
            op = pending[-1]
            if op in seen:
                pending.pop()
                continue
            incomings = set()
            for vs in op._inputs.values():
                if vs not in avail and vs.parent is not None:
                    incomings.add(vs.parent)
            if not incomings:
                avail |= set(op._outputs.values())
                pending.pop()
                res.append(op)
                seen.add(op)
            else:
                pending.extend(incomings)
        return res