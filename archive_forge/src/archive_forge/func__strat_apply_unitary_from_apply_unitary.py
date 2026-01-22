import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def _strat_apply_unitary_from_apply_unitary(unitary_value: Any, args: ApplyUnitaryArgs) -> Optional[np.ndarray]:
    func = getattr(unitary_value, '_apply_unitary_', None)
    if func is None:
        return NotImplemented
    if args.slices is None:
        op_qid_shape = qid_shape_protocol.qid_shape(unitary_value, (2,) * len(args.axes))
        slices = tuple((slice(0, size) for size in op_qid_shape))
    else:
        slices = args.slices
    sub_args = args._for_operation_with_qid_shape(range(len(slices)), slices)
    sub_result = func(sub_args)
    if sub_result is NotImplemented or sub_result is None:
        return sub_result
    return _incorporate_result_into_target(args, sub_args, sub_result)