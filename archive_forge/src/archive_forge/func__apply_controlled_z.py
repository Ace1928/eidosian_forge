from copy import copy
from typing import Sequence, Callable
import pennylane as qml
from pennylane import PauliX, Hadamard, MultiControlledX, CZ, adjoint
from pennylane.wires import Wires
from pennylane.templates import QFT
from pennylane.transforms.core import transform
def _apply_controlled_z(wires, control_wire, work_wires):
    """Provides the circuit to apply a controlled version of the :math:`Z` gate defined in
    `this <https://arxiv.org/abs/1805.00109>`__ paper.

    The multi-qubit gate :math:`Z = I - 2|0\\rangle \\langle 0|` can be performed using the
    conventional multi-controlled-Z gate with an additional bit flip on each qubit before and after.

    This function returns the multi-controlled-Z gate via a multi-controlled-X gate by picking an
    arbitrary target wire to perform the X and adding a Hadamard on that wire either side of the
    transformation.

    Additional control from ``control_wire`` is then included within the multi-controlled-X gate.

    Args:
        wires (Wires): the wires on which the Z gate is applied
        control_wire (Wires): the control wire from the register of phase estimation qubits
        work_wires (Wires): the work wires used in the decomposition
    """
    target_wire = wires[0]
    updated_operations = []
    updated_operations.append(PauliX(target_wire))
    updated_operations.append(Hadamard(target_wire))
    control_values = '0' * (len(wires) - 1) + '1'
    control_wires = wires[1:] + control_wire
    updated_operations.append(MultiControlledX(wires=[*control_wires, target_wire], control_values=control_values, work_wires=work_wires))
    updated_operations.append(Hadamard(target_wire))
    updated_operations.append(PauliX(target_wire))
    return updated_operations