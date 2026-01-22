from typing import Any, cast, Iterable, Optional, Tuple, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.mixture_protocol import mixture
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def _apply_unitary_strat(val: Any, args: 'ApplyMixtureArgs', is_density_matrix: bool) -> Optional[np.ndarray]:
    """Attempt to use `apply_unitary` and return the result.

    If `val` does not support `apply_unitary` returns None.
    """
    left_args = ApplyUnitaryArgs(target_tensor=args.target_tensor, available_buffer=args.auxiliary_buffer0, axes=args.left_axes)
    left_result = apply_unitary(val, left_args, None)
    if left_result is None:
        return None
    if not is_density_matrix:
        return left_result
    right_args = ApplyUnitaryArgs(target_tensor=np.conjugate(left_result), available_buffer=args.auxiliary_buffer0, axes=cast(Tuple[int], args.right_axes))
    right_result = apply_unitary(val, right_args)
    np.conjugate(right_result, out=right_result)
    return right_result