from typing import (
from cirq import protocols, value
from cirq.ops import (
def with_observable(self, observable: Union['cirq.BaseDensePauliString', Iterable['cirq.Pauli']]) -> 'PauliMeasurementGate':
    """Creates a pauli measurement gate with the new observable and same key."""
    if (observable if isinstance(observable, dps.BaseDensePauliString) else dps.DensePauliString(observable)) == self._observable:
        return self
    return PauliMeasurementGate(observable, key=self.key)