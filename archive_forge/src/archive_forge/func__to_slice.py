import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def _to_slice(subspace_def: Tuple[int, ...]):
    if len(subspace_def) < 1:
        raise ValueError(f'Subspace {subspace_def} has zero dimensions.')
    if len(subspace_def) == 1:
        return slice(subspace_def[0], subspace_def[0] + 1, 1)
    step = subspace_def[1] - subspace_def[0]
    for i in range(len(subspace_def) - 1):
        if subspace_def[i + 1] - subspace_def[i] != step:
            raise ValueError(f'Subspace {subspace_def} does not have consistent step size.')
    stop = subspace_def[-1] + step
    return slice(subspace_def[0], stop if stop >= 0 else None, step)