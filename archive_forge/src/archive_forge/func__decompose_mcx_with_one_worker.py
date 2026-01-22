from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _decompose_mcx_with_one_worker(control_wires, target_wire, work_wire):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.3 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a single work wire"""
    tot_wires = len(control_wires) + 2
    partition = int(np.ceil(tot_wires / 2))
    first_part = control_wires[:partition]
    second_part = control_wires[partition:]
    gates = [qml.ctrl(qml.X(work_wire), control=first_part, work_wires=second_part + target_wire), qml.ctrl(qml.X(target_wire), control=second_part + work_wire, work_wires=first_part), qml.ctrl(qml.X(work_wire), control=first_part, work_wires=second_part + target_wire), qml.ctrl(qml.X(target_wire), control=second_part + work_wire, work_wires=first_part)]
    return gates