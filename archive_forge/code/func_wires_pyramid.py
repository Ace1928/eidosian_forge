import pennylane as qml
from pennylane.wires import Wires
def wires_pyramid(wires):
    """Wire sequence for the pyramid pattern."""
    sequence = []
    for layer in range(len(wires) // 2):
        block = wires[layer:len(wires) - layer]
        sequence += [block.subset([i, i + 1]) for i in range(0, len(block) - 1, 2)]
    return sequence