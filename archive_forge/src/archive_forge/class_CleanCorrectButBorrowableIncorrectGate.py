import cirq
import pytest
import numpy as np
class CleanCorrectButBorrowableIncorrectGate(cirq.Gate):
    """Ancilla type determines if the decomposition is correct or not."""

    def __init__(self, use_clean_ancilla: bool) -> None:
        self.ancillas_are_clean = use_clean_ancilla

    def _num_qubits_(self):
        return 2

    def _decompose_with_context_(self, qubits, *, context):
        if self.ancillas_are_clean:
            anc = context.qubit_manager.qalloc(1)
        else:
            anc = context.qubit_manager.qborrow(1)
        yield cirq.CCNOT(*qubits, *anc)
        yield cirq.Z(*anc)
        yield cirq.CCNOT(*qubits, *anc)
        context.qubit_manager.qfree(anc)