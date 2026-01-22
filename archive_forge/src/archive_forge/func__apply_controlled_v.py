from copy import copy
from typing import Sequence, Callable
import pennylane as qml
from pennylane import PauliX, Hadamard, MultiControlledX, CZ, adjoint
from pennylane.wires import Wires
from pennylane.templates import QFT
from pennylane.transforms.core import transform
def _apply_controlled_v(target_wire, control_wire):
    """Provides the circuit to apply a controlled version of the :math:`V` gate defined in
    `this <https://arxiv.org/abs/1805.00109>`__ paper.

    The :math:`V` gate is simply a Pauli-Z gate applied to the ``target_wire``, i.e., the ancilla
    wire in which the expectation value is encoded.

    The controlled version of this gate is then a CZ gate.

    Args:
        target_wire (Wires): the ancilla wire in which the expectation value is encoded
        control_wire (Wires): the control wire from the register of phase estimation qubits
    """
    return [CZ(wires=[control_wire[0], target_wire[0]])]