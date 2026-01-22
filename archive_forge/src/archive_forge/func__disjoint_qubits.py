from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
def _disjoint_qubits(op1: 'cirq.Operation', op2: 'cirq.Operation') -> bool:
    """Returns true only if the operations have qubits in common."""
    return not set(op1.qubits) & set(op2.qubits)