from typing import List, Union, Sequence
import warnings
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.typing import TensorLike
from pennylane.ops import functions
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian
dy/dt = -i H(t) y