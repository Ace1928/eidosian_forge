from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
def to_circuit(self) -> circuit.Circuit:
    return circuit.Circuit(self.all_operations(), strategy=circuit.InsertStrategy.EARLIEST)