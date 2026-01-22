import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class MultiQubitOp(cirq.Operation):
    """Multi-qubit operation with unitary.

        Used to verify that `is_supported_operation` does not attempt to
        allocate the unitary for multi-qubit operations.
        """

    @property
    def qubits(self):
        return cirq.LineQubit.range(100)

    def with_qubits(self, *new_qubits):
        raise NotImplementedError()

    def _has_unitary_(self):
        return True

    def _unitary_(self):
        assert False