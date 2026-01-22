import warnings
from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.typing import TensorLike
from pennylane.wires import Wires
@dataclass
class TransmonSettings:
    """Dataclass that contains the information of a Transmon setup.

    .. see-also:: :func:`transmon_interaction`

    Args:
            connections (List): List `[[idx_q0, idx_q1], ..]` of connected qubits (wires)
            qubit_freq (List[float, Callable]):
            coupling (List[list, TensorLike, Callable]):
            anharmonicity (List[float, Callable]):

    """
    connections: List
    qubit_freq: Union[float, Callable]
    coupling: Union[list, TensorLike, Callable]
    anharmonicity: Union[float, Callable]

    def __eq__(self, other):
        return qml.math.all(self.connections == other.connections) and qml.math.all(self.qubit_freq == other.qubit_freq) and qml.math.all(self.coupling == other.coupling) and qml.math.all(self.anharmonicity == other.anharmonicity)

    def __add__(self, other):
        if other is not None:
            new_connections = list(self.connections) + list(other.connections)
            new_qubit_freq = list(self.qubit_freq) + list(other.qubit_freq)
            new_coupling = list(self.coupling) + list(other.coupling)
            new_anh = list(self.anharmonicity) + list(other.anharmonicity)
            return TransmonSettings(new_connections, new_qubit_freq, new_coupling, anharmonicity=new_anh)
        return self