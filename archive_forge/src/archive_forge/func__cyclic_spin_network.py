import logging
import numpy as np
def _cyclic_spin_network(num_qubits: int, depth: int) -> np.ndarray:
    """
    Same as in the spin-like network, but the first and the last qubits are also connected.

    Args:
        num_qubits: number of qubits.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        A matrix of size ``2 x L`` that defines layers in qubit network.
    """
    cnots = np.zeros((2, depth), dtype=int)
    z = 0
    while True:
        for i in range(0, num_qubits, 2):
            if i + 1 <= num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = i + 1
                z += 1
            if z >= depth:
                return cnots
        for i in range(1, num_qubits, 2):
            if i + 1 <= num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = i + 1
                z += 1
            elif i == num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = 0
                z += 1
            if z >= depth:
                return cnots