from typing import List
from numpy import sqrt
import numpy as np
import cirq
@staticmethod
def triangular_lattice(l: int, x0: float=0, y0: float=0):
    """Returns a triangular lattice of TwoDQubits.

        Args:
            l: Number of qubits along one direction.
            x0: x-coordinate of the first qubit.
            y0: y-coordinate of the first qubit.

        Returns:
            A list of TwoDQubits filling in a triangular lattice.
        """
    coords = np.array([[x, y] for x in range(l + 1) for y in range(l + 1)], dtype=float)
    coords[:, 0] += 0.5 * np.mod(coords[:, 1], 2)
    coords[:, 1] *= np.sqrt(3) / 2
    coords += [x0, y0]
    return [TwoDQubit(coord[0], coord[1]) for coord in coords]