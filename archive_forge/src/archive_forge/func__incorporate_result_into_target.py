import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def _incorporate_result_into_target(args: 'ApplyUnitaryArgs', sub_args: 'ApplyUnitaryArgs', sub_result: np.ndarray):
    """Takes the result of calling `_apply_unitary_` on `sub_args` and
    copies it back into `args.target_tensor` or `args.available_buffer` as
    necessary to return the result of applying the unitary to the full args.
    Also swaps the buffers so the result is always in `args.target_tensor`.

    Args:
        args: The original args.
        sub_args: A version of `args` with transposed and sliced views of
            it's tensors.
        sub_result: The result of calling an object's `_apply_unitary_`
            method on `sub_args`.  A transposed subspace of the desired
            result.

    Returns:
        The full result tensor after applying the unitary.  Always
        `args.target_tensor`.

    Raises:
        ValueError: If `sub_args` tensors are not views of `args` tensors.

    """
    if not (np.may_share_memory(args.target_tensor, sub_args.target_tensor) and np.may_share_memory(args.available_buffer, sub_args.available_buffer)):
        raise ValueError('sub_args.target_tensor and subargs.available_buffer must be views of args.target_tensor and args.available_buffer respectively.')
    is_subspace = sub_args.target_tensor.size < args.target_tensor.size
    if sub_result is sub_args.target_tensor:
        return args.target_tensor
    if sub_result is sub_args.available_buffer:
        if is_subspace:
            sub_args.target_tensor[...] = sub_result
            return args.target_tensor
        return args.available_buffer
    if np.may_share_memory(sub_args.target_tensor, sub_result):
        if is_subspace:
            args.available_buffer[...] = args.target_tensor
        sub_args.available_buffer[...] = sub_result
        return args.available_buffer
    sub_args.target_tensor[...] = sub_result
    return args.target_tensor