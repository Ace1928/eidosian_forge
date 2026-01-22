from typing import Any, Dict, TYPE_CHECKING
from cirq import value, protocols
from cirq._doc import document
from cirq.devices import device
@value.value_equality()
class _UnconstrainedDevice(device.Device):
    """A device that allows everything, infinitely fast."""

    def duration_of(self, operation: 'cirq.Operation') -> 'cirq.Duration':
        return value.Duration(picos=0)

    def validate_moment(self, moment) -> None:
        pass

    def validate_circuit(self, circuit) -> None:
        pass

    def __repr__(self) -> str:
        return 'cirq.UNCONSTRAINED_DEVICE'

    def _value_equality_values_(self) -> Any:
        return ()

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, [])