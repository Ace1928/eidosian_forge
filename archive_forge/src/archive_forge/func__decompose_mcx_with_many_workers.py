from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _decompose_mcx_with_many_workers(control_wires, target_wire, work_wires):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a suitably large register of
    work wires"""
    num_work_wires_needed = len(control_wires) - 2
    work_wires = work_wires[:num_work_wires_needed]
    work_wires_reversed = list(reversed(work_wires))
    control_wires_reversed = list(reversed(control_wires))
    gates = []
    for i in range(len(work_wires)):
        ctrl1 = control_wires_reversed[i]
        ctrl2 = work_wires_reversed[i]
        t = target_wire if i == 0 else work_wires_reversed[i - 1]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))
    gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))
    for i in reversed(range(len(work_wires))):
        ctrl1 = control_wires_reversed[i]
        ctrl2 = work_wires_reversed[i]
        t = target_wire if i == 0 else work_wires_reversed[i - 1]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))
    for i in range(len(work_wires) - 1):
        ctrl1 = control_wires_reversed[i + 1]
        ctrl2 = work_wires_reversed[i + 1]
        t = work_wires_reversed[i]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))
    gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))
    for i in reversed(range(len(work_wires) - 1)):
        ctrl1 = control_wires_reversed[i + 1]
        ctrl2 = work_wires_reversed[i + 1]
        t = work_wires_reversed[i]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))
    return gates