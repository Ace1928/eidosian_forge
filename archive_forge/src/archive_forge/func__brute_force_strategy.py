from typing import Any, Dict, List, Optional, Set, Sequence, Tuple, TYPE_CHECKING
import itertools
import networkx as nx
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper
@classmethod
def _brute_force_strategy(cls, mm: mapping_manager.MappingManager, two_qubit_ops_ints: Sequence[Sequence[QidIntPair]], timestep: int) -> Tuple[QidIntPair, ...]:
    """Inserts SWAPS along the shortest path of the qubits that are the farthest.

        Since swaps along the shortest path are being executed one after the other, in order
        to achieve the physical swaps (M[q1], M[q2]), (M[q2], M[q3]), ..., (M[q_{i-1}], M[q_i]),
        we must execute the logical swaps (q1, q2), (q1, q3), ..., (q_1, qi).
        """
    furthest_op = max(two_qubit_ops_ints[timestep], key=lambda op: mm.dist_on_device(*op))
    path = mm.shortest_path(*furthest_op)
    return tuple([(path[0], path[i + 1]) for i in range(len(path) - 2)])