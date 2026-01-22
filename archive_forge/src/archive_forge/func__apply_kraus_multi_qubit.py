from typing import Any, Iterable, Optional, Sequence, TypeVar, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.kraus_protocol import kraus
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def _apply_kraus_multi_qubit(kraus: Union[Tuple[Any], Sequence[Any]], args: 'ApplyChannelArgs') -> np.ndarray:
    """Use numpy's einsum to apply a multi-qubit channel."""
    qid_shape = tuple((args.target_tensor.shape[i] for i in args.left_axes))
    for kraus_op in kraus:
        np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer0)
        kraus_tensor = np.reshape(kraus_op.astype(args.target_tensor.dtype), qid_shape * 2)
        linalg.targeted_left_multiply(kraus_tensor, args.target_tensor, args.left_axes, out=args.auxiliary_buffer1)
        linalg.targeted_left_multiply(np.conjugate(kraus_tensor), args.auxiliary_buffer1, args.right_axes, out=args.target_tensor)
        args.out_buffer += args.target_tensor
    return args.out_buffer