import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class HasApplyMutateInline:

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
        one = args.subspace_index(1)
        args.target_tensor[one] *= -1
        return args.target_tensor