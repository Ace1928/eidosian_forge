from typing import (
import numbers
import sympy
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import (
def _to_z_basis_ops(self, qubits: Sequence['cirq.Qid']) -> Iterator[raw_types.Operation]:
    """Returns operations to convert the qubits to the computational basis."""
    return self.dense_pauli_string.on(*qubits).to_z_basis_ops()