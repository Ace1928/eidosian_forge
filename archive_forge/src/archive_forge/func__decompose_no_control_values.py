import warnings
import functools
from copy import copy
from functools import wraps
from inspect import signature
from typing import List
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import operation
from pennylane import math as qmlmath
from pennylane.operation import Operator
from pennylane.wires import Wires
from pennylane.compiler import compiler
from .symbolicop import SymbolicOp
from .controlled_decompositions import ctrl_decomp_bisect, ctrl_decomp_zyz
def _decompose_no_control_values(op: Controlled) -> List['operation.Operator']:
    """Decompose without considering control values. Returns None if no decomposition."""
    decomp = _decompose_custom_ops(op)
    if decomp is not None:
        return decomp
    if _is_single_qubit_special_unitary(op.base):
        if len(op.control_wires) >= 2 and qmlmath.get_interface(*op.data) == 'numpy':
            return ctrl_decomp_bisect(op.base, op.control_wires)
        return ctrl_decomp_zyz(op.base, op.control_wires)
    if not op.base.has_decomposition:
        return None
    base_decomp = op.base.decomposition()
    if len(base_decomp) == 0 and isinstance(op.base, qml.GlobalPhase):
        warnings.warn('Controlled-GlobalPhase currently decomposes to nothing, and this will likely produce incorrect results. Consider implementing your circuit with a different set of operations, or use a device that natively supports GlobalPhase.', UserWarning)
    return [ctrl(newop, op.control_wires, work_wires=op.work_wires) for newop in base_decomp]