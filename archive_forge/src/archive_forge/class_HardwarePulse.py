from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from .parametrized_hamiltonian import ParametrizedHamiltonian
@dataclass
class HardwarePulse:
    """Dataclass that contains the information of a single drive pulse. This class is used
    internally in PL to group into a single object all the data related to a single EM field.
    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude of an EM
            field
        phase (Union[float, Callable]): float containing the phase (in radians) of the EM field
        frequency (Union[float, Callable]): float or callable returning the frequency of a
            EM field. In the case of superconducting transmon systems this is the drive frequency.
            In the case of neutral atom rydberg systems this is the detuning between the drive frequency
            and energy gap.
        wires (Union[int, List[int]]): integer or list containing wire values that the EM field
            acts on
    """
    amplitude: Union[float, Callable]
    phase: Union[float, Callable]
    frequency: Union[float, Callable]
    wires: List[Wires]

    def __post_init__(self):
        self.wires = Wires(self.wires)

    def __eq__(self, other):
        return self.amplitude == other.amplitude and self.phase == other.phase and (self.frequency == other.frequency) and (self.wires == other.wires)