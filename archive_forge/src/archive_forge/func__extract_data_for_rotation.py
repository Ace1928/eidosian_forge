from typing import Optional, Callable, List, Union
from functools import reduce
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library.standard_gates import HGate
from ..n_local.n_local import NLocal
def _extract_data_for_rotation(self, pauli, x):
    where_non_i = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
    x = np.asarray(x)
    return x[where_non_i]