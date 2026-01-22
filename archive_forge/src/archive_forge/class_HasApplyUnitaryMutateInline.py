from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
class HasApplyUnitaryMutateInline:

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
        one = args.subspace_index(1)
        args.target_tensor[one] *= 1j
        return args.target_tensor