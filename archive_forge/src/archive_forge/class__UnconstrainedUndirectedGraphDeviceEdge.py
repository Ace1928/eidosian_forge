import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
class _UnconstrainedUndirectedGraphDeviceEdge(UndirectedGraphDeviceEdge):
    """A device edge that allows everything."""

    def duration_of(self, operation: ops.Operation) -> value.Duration:
        return value.Duration(picos=0)

    def validate_operation(self, operation: ops.Operation) -> None:
        pass

    def __eq__(self, other):
        return self.__class__ == other.__class__