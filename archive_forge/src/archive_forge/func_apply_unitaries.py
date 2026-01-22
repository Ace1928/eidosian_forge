import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def apply_unitaries(unitary_values: Iterable[Any], qubits: Sequence['cirq.Qid'], args: Optional[ApplyUnitaryArgs]=None, default: Any=RaiseTypeErrorIfNotProvided) -> Optional[np.ndarray]:
    """Apply a series of unitaries onto a state tensor.

    Uses `cirq.apply_unitary` on each of the unitary values, to apply them to
    the state tensor from the `args` argument.

    CAUTION: if one of the given unitary values does not have a unitary effect,
    forcing the method to terminate, the method will not rollback changes
    from previous unitary values.

    Args:
        unitary_values: The values with unitary effects to apply to the target.
        qubits: The qubits that will be targeted by the unitary values. These
            qubits match up, index by index, with the `indices` property of the
            `args` argument.
        args: A mutable `cirq.ApplyUnitaryArgs` object describing the target
            tensor, available workspace, and axes to operate on. The attributes
            of this object will be mutated as part of computing the result. If
            not specified, this defaults to the zero state of the given qubits
            with an axis ordering matching the given qubit ordering.
        default: What should be returned if any of the unitary values actually
            don't have a unitary effect. If not specified, a TypeError is
            raised instead of returning a default value.

    Returns:
        If any of the unitary values do not have a unitary effect, the
        specified default value is returned (or a TypeError is raised).
        CAUTION: If this occurs, the contents of `args.target_tensor`
        and `args.available_buffer` may have been mutated.

        If all of the unitary values had a unitary effect that was
        successfully applied, this method returns the `np.ndarray`
        storing the final result. This `np.ndarray` may be
        `args.target_tensor`, `args.available_buffer`, or some
        other instance. The caller is responsible for dealing with
        this potential aliasing of the inputs and the result.

    Raises:
        TypeError: An item from `unitary_values` doesn't have a unitary effect
            and `default` wasn't specified.
        ValueError: If the number of qubits does not match the number of
            axes provided in the `args`.
    """
    if args is None:
        qid_shape = qid_shape_protocol.qid_shape(qubits)
        args = ApplyUnitaryArgs.default(qid_shape=qid_shape)
    if len(qubits) != len(args.axes):
        raise ValueError('len(qubits) != len(args.axes)')
    qubit_map = {q.with_dimension(1): args.axes[i] for i, q in enumerate(qubits)}
    state = args.target_tensor
    buffer = args.available_buffer
    for op in unitary_values:
        indices = [qubit_map[q.with_dimension(1)] for q in op.qubits]
        result = apply_unitary(unitary_value=op, args=ApplyUnitaryArgs(state, buffer, indices), default=None)
        if result is None:
            if default is RaiseTypeErrorIfNotProvided:
                raise TypeError(f'cirq.apply_unitaries failed. There was a non-unitary value in the `unitary_values` list.\n\nnon-unitary value type: {type(op)}\nnon-unitary value: {op!r}')
            return default
        if result is buffer:
            buffer = state
        state = result
    return state