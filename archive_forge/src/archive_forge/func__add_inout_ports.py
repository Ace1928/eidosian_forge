import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
def _add_inout_ports(self, before_block, after_block, builder):
    outgoing_nodename = f'outgoing_{before_block.name}'
    outgoing_node = builder.node_maker.make_node(kind='ports', ports=list(before_block.outgoing_states), data=dict(body='outgoing'))
    builder.graph.add_node(outgoing_nodename, outgoing_node)
    incoming_nodename = f'incoming_{after_block.name}'
    incoming_node = builder.node_maker.make_node(kind='ports', ports=list(after_block.incoming_states), data=dict(body='incoming'))
    builder.graph.add_node(incoming_nodename, incoming_node)
    builder.graph.add_edge(incoming_nodename, outgoing_nodename, kind='meta')