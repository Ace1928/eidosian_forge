import logging
import numpy as np
def _cartan_network(num_qubits: int) -> np.ndarray:
    """
    Cartan decomposition in a recursive way, starting from n = 3.

    Args:
        num_qubits: number of qubits.

    Returns:
        2xN matrix that defines layers in qubit network, where N is the
             depth of Cartan decomposition.

    Raises:
        ValueError: if number of qubits is less than 3.
    """
    n = num_qubits
    if n > 3:
        cnots = np.asarray([[0, 0, 0], [1, 1, 1]])
        mult = np.asarray([[n - 2, n - 3, n - 2, n - 3], [n - 1, n - 1, n - 1, n - 1]])
        for _ in range(n - 2):
            cnots = np.hstack((np.tile(np.hstack((cnots, mult)), 3), cnots))
            mult[0, -1] -= 1
            mult = np.tile(mult, 2)
    elif n == 3:
        cnots = np.asarray([[0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1]])
    else:
        raise ValueError(f'The number of qubits must be >= 3, got {n}.')
    return cnots