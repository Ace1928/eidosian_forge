from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def _check_wires(op):
    """Check that wires are a ``Wires`` class and can be mapped."""
    assert isinstance(op.wires, qml.wires.Wires), 'wires must be a wires instance'
    wire_map = {w: ascii_lowercase[i] for i, w in enumerate(op.wires)}
    mapped_op = op.map_wires(wire_map)
    new_wires = qml.wires.Wires(list(ascii_lowercase[:len(op.wires)]))
    assert mapped_op.wires == new_wires, 'wires must be mappable with map_wires'