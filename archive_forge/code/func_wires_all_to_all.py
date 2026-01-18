import pennylane as qml
from pennylane.wires import Wires
def wires_all_to_all(wires):
    """Wire sequence for the all-to-all pattern"""
    sequence = []
    for i in range(len(wires)):
        for j in range(i + 1, len(wires)):
            sequence += [wires.subset([i, j])]
    return sequence