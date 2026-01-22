from typing import Any, FrozenSet, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import measurement_key_protocol
from cirq.type_workarounds import NotImplementedType
Return the keys for controls referenced by the receiving object.

        Returns:
            The measurement keys the value is controlled by. If the value is not
            classically controlled, the result is the empty tuple.
        