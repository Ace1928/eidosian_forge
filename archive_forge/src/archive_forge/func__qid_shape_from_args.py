from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def _qid_shape_from_args(num_qubits: Optional[int], qid_shape: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Returns either `(2,) * num_qubits` or `qid_shape`.

    Raises:
        ValueError: If both arguments are None or their values disagree.
    """
    if num_qubits is None and qid_shape is None:
        raise ValueError('Either the num_qubits or qid_shape argument must be specified. Both were None.')
    if num_qubits is None:
        return cast(Tuple[int, ...], qid_shape)
    if qid_shape is None:
        return (2,) * num_qubits
    if len(qid_shape) != num_qubits:
        raise ValueError(f'num_qubits != len(qid_shape). num_qubits was {num_qubits!r}. qid_shape was {qid_shape!r}.')
    return qid_shape