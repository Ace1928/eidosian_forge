import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def _get_active_pairs(graph: nx.Graph, grid_layer: GridInteractionLayer):
    """Extract pairs of qubits from a device graph and a GridInteractionLayer."""
    for edge in graph.edges:
        if edge in grid_layer:
            yield edge