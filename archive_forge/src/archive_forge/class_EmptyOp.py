import numpy as np
import pytest
import cirq
class EmptyOp(cirq.Operation):
    """A trivial operation that will be recognized as `_apply_unitary_`-able."""

    @property
    def qubits(self):
        return ()

    def with_qubits(self, *new_qubits):
        return self