import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
class IgnoreAxisEffect:

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
        if args.target_tensor.shape[0] > 1:
            args.available_buffer[0] = args.target_tensor[1]
            args.available_buffer[1] = args.target_tensor[0]
        return args.available_buffer

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])

    def _num_qubits_(self):
        return 1