import logging
import numpy as np
def _spin_network(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a spin-like network.

    Args:
        num_qubits: number of qubits.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        A matrix of size ``2 x L`` that defines layers in qubit network.
    """
    layer = 0
    cnots = np.zeros((2, depth), dtype=int)
    while True:
        for i in range(0, num_qubits - 1, 2):
            cnots[0, layer] = i
            cnots[1, layer] = i + 1
            layer += 1
            if layer >= depth:
                return cnots
        for i in range(1, num_qubits - 1, 2):
            cnots[0, layer] = i
            cnots[1, layer] = i + 1
            layer += 1
            if layer >= depth:
                return cnots