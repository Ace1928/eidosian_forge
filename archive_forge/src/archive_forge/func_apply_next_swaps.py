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
def apply_next_swaps(self, require_frontier_adjacency: bool=False):
    """Applies a few SWAPs to get the mapping closer to one in which the
        next logical gates can be applied.

        See route_circuit_greedily for more details.
        """
    time_slices = get_time_slices(self.remaining_dag)
    if require_frontier_adjacency:
        frontier_edges = sorted(time_slices[0].edges)
        self.bring_farthest_pair_together(frontier_edges)
        return
    for k in range(1, self.max_search_radius + 1):
        candidate_swap_sets = list(self.get_edge_sets(k))
        for time_slice in time_slices:
            edges = sorted(time_slice.edges)
            distance_vectors = list((self.get_distance_vector(edges, swap_set) for swap_set in candidate_swap_sets))
            dominated_indices = _get_dominated_indices(distance_vectors)
            candidate_swap_sets = [S for i, S in enumerate(candidate_swap_sets) if i not in dominated_indices]
            if len(candidate_swap_sets) == 1:
                self.apply_swap(*candidate_swap_sets[0])
                if list(self.remaining_dag.findall_nodes_until_blocked(self.acts_on_nonadjacent_qubits)):
                    return
                else:
                    break
    self.apply_next_swaps(True)