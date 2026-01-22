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
def _handle_pauli_x_based_controlled_ops(op, control, control_values, work_wires):
    """Handles PauliX-based controlled operations."""
    op_map = {(qml.PauliX, 1): qml.CNOT, (qml.PauliX, 2): qml.Toffoli, (qml.CNOT, 1): qml.Toffoli}
    custom_key = (type(op), len(control))
    if custom_key in op_map and all(control_values):
        qml.QueuingManager.remove(op)
        return op_map[custom_key](wires=control + op.wires)
    if isinstance(op, qml.PauliX):
        return qml.MultiControlledX(wires=control + op.wires, control_values=control_values, work_wires=work_wires)
    work_wires = work_wires or []
    return qml.MultiControlledX(wires=control + op.wires, control_values=control_values + op.control_values, work_wires=work_wires + op.work_wires)