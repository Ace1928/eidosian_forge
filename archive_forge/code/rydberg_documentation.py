from dataclasses import dataclass
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian, HardwarePulse, drive
from pennylane.wires import Wires
from pennylane.pulse.hardware_hamiltonian import _reorder_parameters
Dataclass that contains the information of a Rydberg setup.

    Args:
        register (list): coordinates of atoms
        interaction_coeff (float): interaction coefficient C6 from C6/(Rij)**6 term in :func:`rydberg_interaction`
    