import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def _apply_unitary_from_matrix(matrix: np.ndarray, unitary_value: Any, args: ApplyUnitaryArgs):
    if args.slices is None:
        val_qid_shape = qid_shape_protocol.qid_shape(unitary_value, default=(2,) * len(args.axes))
        slices = tuple((slice(0, size) for size in val_qid_shape))
    else:
        slices = args.slices
        val_qid_shape = tuple((((s.step if s.stop is None else s.stop) - s.start) // (s.step or 1) for s in slices))
    sub_args = args._for_operation_with_qid_shape(range(len(slices)), slices)
    matrix = matrix.astype(sub_args.target_tensor.dtype)
    if len(val_qid_shape) == 1 and val_qid_shape[0] <= 2:
        subspaces = [(..., level) for level in range(val_qid_shape[0])]
        sub_result = linalg.apply_matrix_to_slices(sub_args.target_tensor, matrix, subspaces, out=sub_args.available_buffer)
    else:
        sub_result = linalg.targeted_left_multiply(matrix.reshape(val_qid_shape * 2), sub_args.target_tensor, sub_args.axes, out=sub_args.available_buffer)
    return _incorporate_result_into_target(args, sub_args, sub_result)