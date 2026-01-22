import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
class SameQuditEffect:

    def _qid_shape_(self):
        return (3,)

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
        args.available_buffer[..., 0] = args.target_tensor[..., 2]
        args.available_buffer[..., 1] = args.target_tensor[..., 0]
        args.available_buffer[..., 2] = args.target_tensor[..., 1]
        return args.available_buffer

    def _unitary_(self):
        return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])