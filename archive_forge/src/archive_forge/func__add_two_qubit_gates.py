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
def _add_two_qubit_gates(graph, vertex, neighbor, type_1, type_2, qubit_1, qubit_2):
    """Return the list of two qubit gates giveeen the vertex and its neighbor."""
    if type_1 == type_2:
        if graph.edge_type(graph.edge(vertex, neighbor)) != EdgeType.HADAMARD:
            raise qml.QuantumFunctionError('Two green or respectively two red nodes connected by a simple edge does not have a circuit representation.')
        if type_1 == VertexType.Z:
            op = qml.CZ(wires=[qubit_2, qubit_1])
            return [op]
        op_1 = qml.Hadamard(wires=qubit_2)
        op_2 = qml.CNOT(wires=[qubit_2, qubit_1])
        op_3 = qml.Hadamard(wires=qubit_2)
        return [op_1, op_2, op_3]
    if graph.edge_type(graph.edge(vertex, neighbor)) != EdgeType.SIMPLE:
        raise qml.QuantumFunctionError('A green and red node connected by a Hadamard edge does not have a circuit representation.')
    op = qml.CNOT(wires=[qubit_1, qubit_2])
    return [op]