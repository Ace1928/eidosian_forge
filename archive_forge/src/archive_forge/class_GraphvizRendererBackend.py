import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
class GraphvizRendererBackend(AbstractRendererBackend):
    """The backend for using graphviz to render a GraphBacking.
    """

    def __init__(self, g=None):
        from graphviz import Digraph
        self.digraph = Digraph() if g is None else g

    def render_node(self, k: str, node: GraphNode):
        if node.kind == 'valuestate':
            self.digraph.node(k, label=node.data['body'], shape='rect')
        elif node.kind == 'op':
            self.digraph.node(k, label=node.data['body'], shape='box', style='rounded')
        elif node.kind == 'effect':
            self.digraph.node(k, label=node.data['body'], shape='circle')
        elif node.kind == 'meta':
            self.digraph.node(k, label=node.data['body'], shape='plain', fontcolor='grey')
        elif node.kind == 'ports':
            ports = [f'<{x}> {x}' for x in node.ports]
            label = f'{node.data['body']} | {'|'.join(ports)}'
            self.digraph.node(k, label=label, shape='record')
        elif node.kind == 'cfg':
            self.digraph.node(k, label=node.data['body'], shape='plain', fontcolor='blue')
        else:
            self.digraph.node(k, label=f'{k}\n{node.kind}\n{node.data.get('body', '')}', shape='rect')

    def render_edge(self, edge: GraphEdge):
        attrs = {}
        if edge.headlabel is not None:
            attrs['headlabel'] = edge.headlabel
        if edge.taillabel is not None:
            attrs['taillabel'] = edge.taillabel
        if edge.kind is not None:
            if edge.kind == 'effect':
                attrs['style'] = 'dotted'
            elif edge.kind == 'meta':
                attrs['style'] = 'invis'
            elif edge.kind == 'cfg':
                attrs['style'] = 'solid'
                attrs['color'] = 'blue'
            else:
                raise ValueError(edge.kind)
        src = str(edge.src)
        dst = str(edge.dst)
        if edge.src_port:
            src += f':{edge.src_port}'
        if edge.dst_port:
            dst += f':{edge.dst_port}'
        self.digraph.edge(src, dst, **attrs)

    @contextmanager
    def render_cluster(self, name: str):
        with self.digraph.subgraph(name=f'cluster_{name}') as subg:
            attrs = dict(color='black', bgcolor='white')
            if name.startswith('regionouter'):
                attrs['bgcolor'] = 'grey'
            elif name.startswith('loop_'):
                attrs['color'] = 'blue'
            elif name.startswith('switch_'):
                attrs['color'] = 'green'
            subg.attr(**attrs)
            yield type(self)(subg)