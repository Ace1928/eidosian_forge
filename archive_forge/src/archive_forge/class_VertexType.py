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
class VertexType:
    """Type of a vertex in the graph.

    This class is copied from PyZX as we do not make PyZX a Pennylane requirement.

    Copyright (C) 2018 - Aleks Kissinger and John van de Wetering"""
    BOUNDARY = 0
    Z = 1
    X = 2
    H_BOX = 3