from typing import Any, cast, Iterable, Optional, Tuple, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.mixture_protocol import mixture
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def _mixture_strat(val: Any, args: 'ApplyMixtureArgs', is_density_matrix: bool) -> np.ndarray:
    """Attempt to use unitary matrices in _mixture_ and return the result."""
    args.out_buffer[:] = 0
    np.copyto(dst=args.auxiliary_buffer1, src=args.target_tensor)
    for prob, op in val:
        np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer1)
        right_result = _apply_unitary_strat(op, args, is_density_matrix)
        if right_result is None:
            right_result = _apply_unitary_from_matrix_strat(op, args, is_density_matrix)
        args.out_buffer += prob * right_result
    return args.out_buffer