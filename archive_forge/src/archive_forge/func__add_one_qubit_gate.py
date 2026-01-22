from functools import partial
from typing import Sequence, Callable
from collections import OrderedDict
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane.transforms import transform
from pennylane.wires import Wires
def _add_one_qubit_gate(param, type_1, qubit_1, decompose_phases):
    """Return the list of one qubit gates, that will be added to the tape."""
    if decompose_phases:
        type_z = type_1 == VertexType.Z
        if type_z and param.denominator == 2:
            op = qml.adjoint(qml.S(wires=qubit_1)) if param.numerator == 3 else qml.S(wires=qubit_1)
            return [op]
        if type_z and param.denominator == 4:
            if param.numerator in (1, 7):
                op = qml.adjoint(qml.T(wires=qubit_1)) if param.numerator == 7 else qml.T(wires=qubit_1)
                return [op]
            if param.numerator in (3, 5):
                op1 = qml.Z(qubit_1)
                op2 = qml.adjoint(qml.T(wires=qubit_1)) if param.numerator == 3 else qml.T(wires=qubit_1)
                return [op1, op2]
        if param == 1:
            op = qml.Z(qubit_1) if type_1 == VertexType.Z else qml.X(qubit_1)
            return [op]
        if param != 0:
            scaled_param = np.pi * float(param)
            op_class = qml.RZ if type_1 == VertexType.Z else qml.RX
            return [op_class(scaled_param, wires=qubit_1)]
    if param != 0:
        scaled_param = np.pi * float(param)
        op_class = qml.RZ if type_1 == VertexType.Z else qml.RX
        return [op_class(scaled_param, wires=qubit_1)]
    return []