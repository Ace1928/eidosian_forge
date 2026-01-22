from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def _hashable_parameters(params):
    """Convert the parameters of a gate into a hashable format for lookup in a dictionary."""
    try:
        hash(params)
        return params
    except TypeError:
        pass
    if isinstance(params, (list, tuple)):
        return tuple((_hashable_parameters(x) for x in params))
    if isinstance(params, np.ndarray):
        return (np.ndarray, params.tobytes())
    return ('fallback', str(params))