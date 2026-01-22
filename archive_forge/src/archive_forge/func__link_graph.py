from numbers import Number
from collections import namedtuple
import numpy as np
import rustworkx as rx
from pennylane.measurements import MeasurementProcess
from pennylane.resource import ResourcesOperation
def _link_graph(target_index, sub_graph, node_index):
    """Link incoming and outgoing edges for the initial node to the sub-graph"""
    if target_index == node_index:
        return sub_graph.nodes().index(f'{node_index}.0')
    return sub_graph.nodes().index(f'{node_index}.1')