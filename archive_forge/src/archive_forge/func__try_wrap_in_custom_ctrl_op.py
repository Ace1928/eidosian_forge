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
def _try_wrap_in_custom_ctrl_op(op, control, control_values=None, work_wires=None):
    """Wraps a controlled operation in custom ControlledOp, returns None if not applicable."""
    ops_with_custom_ctrl_ops = _get_special_ops()
    custom_key = (type(op), len(control))
    if custom_key in ops_with_custom_ctrl_ops and all(control_values):
        qml.QueuingManager.remove(op)
        return ops_with_custom_ctrl_ops[custom_key](*op.data, control + op.wires)
    if isinstance(op, qml.QubitUnitary):
        return qml.ControlledQubitUnitary(op, control_wires=control, control_values=control_values, work_wires=work_wires)
    return None