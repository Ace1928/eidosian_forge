from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, TYPE_CHECKING
from cirq.ops import raw_types
Transforms the Basis so that it applies to wrapped qubits.

        Args:
            externalize: Converts an internal qubit understood by the underlying
                basis into an external qubit understood by the caller.
            internalize: Converts an external qubit understood by the caller
                into an internal qubit understood by the underlying basis.

        Returns:
            A basis that transforms qubits understood by the caller into qubits
            understood by an underlying basis, uses that to order the qubits,
            then wraps the ordered qubits back up for the caller.
        