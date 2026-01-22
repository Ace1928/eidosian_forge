from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
class HasApplyUnitaryOutputInBuffer:

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
        zero = args.subspace_index(0)
        one = args.subspace_index(1)
        args.available_buffer[zero] = args.target_tensor[zero]
        args.available_buffer[one] = 1j * args.target_tensor[one]
        return args.available_buffer