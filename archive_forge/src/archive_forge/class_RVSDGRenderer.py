import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
class RVSDGRenderer(RegionVisitor):
    """Convert a RVSDG into a GraphBacking
    """

    def visit_block(self, block: BasicBlock, builder: GraphBuilder):
        nodename = block.name
        node_maker = builder.node_maker.subgroup(f'metaregion_{nodename}')
        if isinstance(block, DDGBlock):
            node = node_maker.make_node(kind='cfg', data=dict(body=nodename))
            builder.graph.add_node(nodename, node)
            block.render_graph(replace(builder, node_maker=node_maker))
        else:
            body = '(tbd)'
            if isinstance(block, DDGBranch):
                body = f'branch_value_table:\n{block.branch_value_table}'
            elif isinstance(block, DDGControlVariable):
                body = f'variable_assignment:\n{block.variable_assignment}'
            node = node_maker.make_node(kind='cfg', data=dict(body=body))
            builder.graph.add_node(nodename, node)
            if isinstance(block, DDGProtocol):
                self._add_inout_ports(block, block, replace(builder, node_maker=node_maker))
        for dstnode in block.jump_targets:
            builder.graph.add_edge(nodename, dstnode, kind='cfg')
        return builder

    def _add_inout_ports(self, before_block, after_block, builder):
        outgoing_nodename = f'outgoing_{before_block.name}'
        outgoing_node = builder.node_maker.make_node(kind='ports', ports=list(before_block.outgoing_states), data=dict(body='outgoing'))
        builder.graph.add_node(outgoing_nodename, outgoing_node)
        incoming_nodename = f'incoming_{after_block.name}'
        incoming_node = builder.node_maker.make_node(kind='ports', ports=list(after_block.incoming_states), data=dict(body='incoming'))
        builder.graph.add_node(incoming_nodename, incoming_node)
        builder.graph.add_edge(incoming_nodename, outgoing_nodename, kind='meta')

    def visit_linear(self, region: RegionBlock, builder: GraphBuilder):
        nodename = region.name
        node_maker = builder.node_maker.subgroup(f'regionouter_{nodename}')
        if isinstance(region, DDGProtocol):
            self._add_inout_ports(region, region, replace(builder, node_maker=node_maker))
        subbuilder = replace(builder, node_maker=node_maker.subgroup(f'{region.kind}_{nodename}'))
        node = node_maker.make_node(kind='cfg', data=dict(body=nodename))
        builder.graph.add_node(region.name, node)
        super().visit_linear(region, subbuilder)
        self._connect_internal(region, builder)
        builder.graph.add_edge(region.name, region.header, kind='cfg')
        return builder

    def _connect_internal(self, region, builder):
        header = region.subregion[region.header]
        if isinstance(region, DDGProtocol) and isinstance(header, DDGProtocol):
            for k in region.incoming_states:
                builder.graph.add_edge(f'incoming_{region.name}', f'incoming_{header.name}', src_port=k, dst_port=k)
        exiting = region.subregion[region.exiting]
        if isinstance(region, DDGProtocol) and isinstance(exiting, DDGProtocol):
            assert isinstance(region, RegionBlock)
            for k in region.outgoing_states & exiting.outgoing_states:
                builder.graph.add_edge(f'outgoing_{exiting.name}', f'outgoing_{region.name}', src_port=k, dst_port=k)

    def visit_graph(self, scfg: SCFG, builder):
        """Overriding"""
        toposorted = self._toposort_graph(scfg)
        label: str
        last_label: str | None = None
        for lvl in toposorted:
            for label in lvl:
                builder = self.visit(scfg[label], builder)
                if last_label is not None:
                    last_node = scfg[last_label]
                    node = scfg[label]
                    self._connect_inout_ports(last_node, node, builder)
                last_label = label
        return builder

    def _connect_inout_ports(self, last_node, node, builder):
        if isinstance(last_node, DDGProtocol) and isinstance(node, DDGProtocol):
            for k in last_node.outgoing_states:
                builder.graph.add_edge(f'outgoing_{last_node.name}', f'incoming_{node.name}', src_port=k, dst_port=k)

    def visit_loop(self, region: RegionBlock, builder: GraphBuilder):
        return self.visit_linear(region, builder)

    def visit_switch(self, region: RegionBlock, builder: GraphBuilder):
        nodename = region.name
        node_maker = builder.node_maker.subgroup(f'regionouter_{nodename}')
        if isinstance(region, DDGProtocol):
            self._add_inout_ports(region, region, replace(builder, node_maker=node_maker))
        subbuilder = replace(builder, node_maker=node_maker.subgroup(f'{region.kind}_{nodename}'))
        node = node_maker.make_node(kind='cfg', data=dict(body=nodename))
        builder.graph.add_node(region.name, node)
        builder.graph.add_edge(region.name, region.header)
        head = region.subregion[region.header]
        tail = region.subregion[region.exiting]
        self.visit_linear(head, subbuilder)
        for blk in region.subregion.graph.values():
            if blk.kind == 'branch':
                self._connect_inout_ports(head, blk, subbuilder)
                self.visit_linear(blk, subbuilder)
                self._connect_inout_ports(blk, tail, subbuilder)
        self.visit_linear(tail, subbuilder)
        self._connect_internal(region, builder)
        return builder

    def render(self, rvsdg: SCFG) -> GraphBacking:
        """Render a RVSDG into a GraphBacking
        """
        builder = GraphBuilder.make()
        self.visit_graph(rvsdg, builder)
        builder.graph.verify()
        return builder.graph