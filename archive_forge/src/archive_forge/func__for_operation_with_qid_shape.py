import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def _for_operation_with_qid_shape(self, indices: Iterable[int], slices: Tuple[Union[int, slice], ...]) -> 'ApplyUnitaryArgs':
    """Creates a sliced and transposed view of `self` appropriate for an
        operation with shape `qid_shape` on qubits with the given indices.

        Example:
            sub_args = args._for_operation_with_qid_shape(indices, (2, 2, 2))
            # Slice where the first qubit is |1>.
            sub_args.target_tensor[..., 1, :, :]

        Args:
            indices: Integer indices into `self.axes` specifying which qubits
                the operation applies to.
            slices: The slices of the operation, the subdimension in each qubit
                the operation applies to.

        Returns: A new `ApplyUnitaryArgs` where `sub_args.target_tensor` and
            `sub_args.available_buffer` are sliced and transposed views of
            `self.target_tensor` and `self.available_buffer` respectively.
        """
    slices = tuple((size if isinstance(size, slice) else slice(0, size) for size in slices))
    sub_axes = [self.axes[i] for i in indices]
    axis_set = set(sub_axes)
    other_axes = [axis for axis in range(len(self.target_tensor.shape)) if axis not in axis_set]
    ordered_axes = (*other_axes, *sub_axes)
    target_tensor = self.target_tensor.transpose(*ordered_axes)[..., *slices]
    available_buffer = self.available_buffer.transpose(*ordered_axes)[..., *slices]
    new_axes = range(len(other_axes), len(ordered_axes))
    return ApplyUnitaryArgs(target_tensor, available_buffer, new_axes)