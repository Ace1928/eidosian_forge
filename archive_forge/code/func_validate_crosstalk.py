import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def validate_crosstalk(self, operation: ops.Operation, other_operations: Iterable[ops.Operation]) -> None:
    adjacent_crosstalk_edges = frozenset(self.crosstalk_graph._adjacency_lists.get(frozenset(operation.qubits), ()))
    for crosstalk_edge in adjacent_crosstalk_edges:
        label = self.crosstalk_graph.labelled_edges[crosstalk_edge]
        validator = raise_crosstalk_error(operation, *other_operations) if label is None else label
        for crosstalk_operations in itertools.combinations(other_operations, len(crosstalk_edge) - 1):
            validator(operation, *crosstalk_operations)