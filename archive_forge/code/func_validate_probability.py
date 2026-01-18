from typing import Any, Sequence, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.type_workarounds import NotImplementedType
def validate_probability(p, p_str):
    if p < 0:
        raise ValueError(f'{p_str} was less than 0.')
    elif p > 1:
        raise ValueError(f'{p_str} was greater than 1.')