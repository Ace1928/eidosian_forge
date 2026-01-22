import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
class UndirectedGraphDevice(devices.Device):
    """A device whose properties are represented by an edge-labelled graph.

    Each (undirected) edge of the device graph is labelled by an
    UndirectedGraphDeviceEdge or None. None indicates that any operation is
    allowed and has zero duration.

    Each (undirected) edge of the constraint graph is labelled either by a
    function or None. The function takes as arguments operations on the
    adjacent device edges and raises an error if they are not simultaneously
    executable. If None, no such operations are allowed.

    Note that
        * the crosstalk graph is allowed to have vertices (i.e. device edges)
            that do not exist in the graph device.
        * duration_of does not check that operation is valid.
    """

    def __init__(self, device_graph: Optional[UndirectedHypergraph]=None, crosstalk_graph: Optional[UndirectedHypergraph]=None) -> None:
        """Inits UndirectedGraphDevice.

        Args:
            device_graph: An undirected hypergraph whose vertices correspond to
                qubits and whose edges determine allowable operations and their
                durations.
            crosstalk_graph: An undirected hypergraph whose vertices are edges
                of device_graph and whose edges give simultaneity constraints
                thereon.

        Raises:
            TypeError: If the crosstalk graph is not a valid crosstalk graph.
        """
        if device_graph is None:
            device_graph = UndirectedHypergraph()
        if not is_undirected_device_graph(device_graph):
            raise TypeError(f'not is_undirected_device_graph({device_graph})')
        if crosstalk_graph is None:
            crosstalk_graph = UndirectedHypergraph()
        if not is_crosstalk_graph(crosstalk_graph):
            raise TypeError(f'not is_crosstalk_graph({crosstalk_graph})')
        self.device_graph = device_graph
        self.crosstalk_graph = crosstalk_graph

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return cast(Tuple['cirq.Qid', ...], tuple(sorted(self.device_graph.vertices)))

    @property
    def edges(self):
        return tuple(sorted(self.device_graph.edges))

    @property
    def labelled_edges(self):
        return self.device_graph.labelled_edges

    def get_device_edge_from_op(self, operation: ops.Operation) -> UndirectedGraphDeviceEdge:
        return self.device_graph.labelled_edges[frozenset(operation.qubits)]

    def duration_of(self, operation: ops.Operation) -> value.Duration:
        return self.get_device_edge_from_op(operation).duration_of(operation)

    def validate_operation(self, operation: ops.Operation) -> None:
        try:
            device_edge = self.get_device_edge_from_op(operation)
        except Exception as error:
            if frozenset(operation.qubits) not in self.device_graph.edges:
                error = ValueError(f'{operation.qubits} not in device graph edges')
            raise error
        device_edge.validate_operation(operation)

    def validate_crosstalk(self, operation: ops.Operation, other_operations: Iterable[ops.Operation]) -> None:
        adjacent_crosstalk_edges = frozenset(self.crosstalk_graph._adjacency_lists.get(frozenset(operation.qubits), ()))
        for crosstalk_edge in adjacent_crosstalk_edges:
            label = self.crosstalk_graph.labelled_edges[crosstalk_edge]
            validator = raise_crosstalk_error(operation, *other_operations) if label is None else label
            for crosstalk_operations in itertools.combinations(other_operations, len(crosstalk_edge) - 1):
                validator(operation, *crosstalk_operations)

    def validate_moment(self, moment: 'cirq.Moment'):
        super().validate_moment(moment)
        ops = moment.operations
        for i, op in enumerate(ops):
            other_ops = ops[:i] + ops[i + 1:]
            self.validate_crosstalk(op, other_ops)

    def __eq__(self, other):
        return self.device_graph == other.device_graph and self.crosstalk_graph == other.crosstalk_graph

    def __iadd__(self, other):
        self.device_graph += other.device_graph
        self.crosstalk_graph += other.crosstalk_graph
        return self

    def __copy__(self):
        return self.__class__(device_graph=self.device_graph.__copy__(), crosstalk_graph=self.crosstalk_graph.__copy__())

    def __add__(self, other):
        device_sum = self.__copy__()
        device_sum += other
        return device_sum