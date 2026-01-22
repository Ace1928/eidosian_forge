import warnings
from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.typing import TensorLike
from pennylane.wires import Wires
def ad(wire, d=2):
    """annihilation operator"""
    return qml.s_prod(0.5, qml.X(wire)) + qml.s_prod(-0.5j, qml.Y(wire))