from typing import (
import numbers
import sympy
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import (
def xor_nonlocal_decompose(qubits: Iterable[raw_types.Qid], onto_qubit: 'cirq.Qid') -> Iterable[raw_types.Operation]:
    """Decomposition ignores connectivity."""
    for qubit in qubits:
        if qubit != onto_qubit:
            yield common_gates.CNOT(qubit, onto_qubit)