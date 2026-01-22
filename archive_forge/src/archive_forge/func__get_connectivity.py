import logging
import numpy as np
def _get_connectivity(num_qubits: int, connectivity: str) -> dict:
    """
    Generates connectivity structure between qubits.

    Args:
        num_qubits: number of qubits.
        connectivity: type of connectivity structure, ``{"full", "line", "star"}``.

    Returns:
        dictionary of allowed links between qubits.

    Raises:
         ValueError: if unsupported type of CNOT-network layout is passed.
    """
    if num_qubits == 1:
        links = {0: [0]}
    elif connectivity == 'full':
        links = {i: list(range(num_qubits)) for i in range(num_qubits)}
    elif connectivity == 'line':
        links = {i: [i - 1, i, i + 1] for i in range(1, num_qubits - 1)}
        links[0] = [0, 1]
        links[num_qubits - 1] = [num_qubits - 2, num_qubits - 1]
    elif connectivity == 'star':
        links = {i: [0, i] for i in range(1, num_qubits)}
        links[0] = list(range(num_qubits))
    else:
        raise ValueError(f'Unknown connectivity type, expects one of {_CONNECTIVITY_TYPES}, got {connectivity}')
    return links