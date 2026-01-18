from typing import Any, TypeVar, Union, Optional
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import ApplyUnitaryArgs, apply_unitaries
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
Whether this value has a unitary matrix representation.

        This method is used by the global `cirq.has_unitary` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using _unitary_ with a default value, or False if neither exist.

        Returns:
            True if the value has a unitary matrix representation, False
            otherwise.
        