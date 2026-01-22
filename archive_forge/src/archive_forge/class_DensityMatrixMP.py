from typing import Sequence, Optional
import pennylane as qml
from pennylane.wires import Wires, WireError
from .measurements import State, StateMeasurement
class DensityMatrixMP(StateMP):
    """Measurement process that returns the quantum state in the computational basis.

    Please refer to :func:`density_matrix` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    def __init__(self, wires: Wires, id: Optional[str]=None):
        super().__init__(wires=wires, id=id)

    def shape(self, device, shots):
        num_shot_elements = sum((s.copies for s in shots.shot_vector)) if shots.has_partitioned_shots else 1
        dim = 2 ** len(self.wires)
        return (dim, dim) if num_shot_elements == 1 else tuple(((dim, dim) for _ in range(num_shot_elements)))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        return qml.math.reduce_statevector(state, indices=mapped_wires)