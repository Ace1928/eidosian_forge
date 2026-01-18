from typing import (
from cirq import protocols, value
from cirq.ops import (
def with_key(self, key: Union[str, 'cirq.MeasurementKey']) -> 'PauliMeasurementGate':
    """Creates a pauli measurement gate with a new key but otherwise identical."""
    if key == self.key:
        return self
    return PauliMeasurementGate(self._observable, key=key)