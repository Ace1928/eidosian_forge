from typing import Callable, Optional, Sequence, Union
import cirq
def _validate_circuit(self, circuits: Sequence[cirq.AbstractCircuit], sweeps: Sequence[cirq.Sweepable], repetitions: Union[int, Sequence[int]]):
    if self._device:
        for circuit in circuits:
            self._device.validate_circuit(circuit)
    if self._validator:
        self._validator(circuits, sweeps, repetitions)