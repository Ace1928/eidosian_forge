import itertools
from typing import (
import numpy as np
import networkx as nx
from cirq import circuits, ops, value
import cirq.contrib.acquaintance as cca
from cirq.contrib import circuitdag
from cirq.contrib.routing.initialization import get_initial_mapping
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import get_time_slices, ops_are_consistent_with_device_graph
def _get_dominated_indices(vectors: List[np.ndarray]) -> Set[int]:
    """Get the indices of vectors that are element-wise at least some other
    vector.
    """
    dominated_indices = set()
    for i, v in enumerate(vectors):
        for w in vectors[:i] + vectors[i + 1:]:
            if all(v >= w):
                dominated_indices.add(i)
                break
    return dominated_indices