import warnings
from typing import Iterable
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag
import pennylane as qml
from pennylane.operation import (
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from pennylane.ops.qubit.parametric_ops_single_qubit import stack_last
from .controlled import ControlledOp
from .controlled_decompositions import decompose_mcx
def _check_and_convert_control_values(control_values, control_wires):
    if isinstance(control_values, str):
        if not set(control_values).issubset({'1', '0'}):
            raise ValueError("String of control values can contain only '0' or '1'.")
        control_values = [int(x) for x in control_values]
    if control_values is None:
        return [1] * len(control_wires)
    if len(control_values) != len(control_wires):
        raise ValueError('Length of control values must equal number of control wires.')
    return control_values