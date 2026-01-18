from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from .parametrized_hamiltonian import ParametrizedHamiltonian
Deals with the special case where a HardwareHamiltonian is added to a
        ParametrizedHamiltonian. Ensures that this returns a HardwareHamiltonian where
        the order of the parametrized coefficients and operators matches the order of
        the hamiltonians, i.e. that
        ParametrizedHamiltonian + HardwareHamiltonian
        returns a HardwareHamiltonian where the call expects params = [params_PH] + [params_RH]
        