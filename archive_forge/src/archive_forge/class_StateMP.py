from typing import Sequence, Optional
import pennylane as qml
from pennylane.wires import Wires, WireError
from .measurements import State, StateMeasurement
class StateMP(StateMeasurement):
    """Measurement process that returns the quantum state in the computational basis.

    Please refer to :func:`state` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    def __init__(self, wires: Optional[Wires]=None, id: Optional[str]=None):
        super().__init__(wires=wires, id=id)

    @property
    def return_type(self):
        return State

    @property
    def numeric_type(self):
        return complex

    def shape(self, device, shots):
        num_shot_elements = sum((s.copies for s in shots.shot_vector)) if shots.has_partitioned_shots else 1
        dim = 2 ** len(self.wires) if self.wires else 2 ** len(device.wires)
        return (dim,) if num_shot_elements == 1 else tuple(((dim,) for _ in range(num_shot_elements)))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        wires = self.wires
        if not wires or wire_order == wires:
            return qml.math.cast(state, 'complex128')
        if set(wires) != set(wire_order):
            raise WireError(f'Unexpected unique wires {Wires.unique_wires([wires, wire_order])} found. Expected wire order {wire_order} to be a rearrangement of {wires}')
        shape = (2,) * len(wires)
        flat_shape = (2 ** len(wires),)
        desired_axes = [wire_order.index(w) for w in wires]
        if qml.math.ndim(state) == 2:
            batch_size = qml.math.shape(state)[0]
            shape = (batch_size,) + shape
            flat_shape = (batch_size,) + flat_shape
            desired_axes = [0] + [i + 1 for i in desired_axes]
        state = qml.math.reshape(state, shape)
        state = qml.math.transpose(state, desired_axes)
        state = qml.math.reshape(state, flat_shape)
        return qml.math.cast(state, 'complex128')