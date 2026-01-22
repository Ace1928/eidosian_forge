from typing import List
from numpy import sqrt
import numpy as np
import cirq
class TwoDQubit(ThreeDQubit):
    """A qubit in 2d."""

    def __init__(self, x: float, y: float):
        super().__init__(x, y, z=0)

    @staticmethod
    def square(diameter: int, x0: float=0, y0: float=0) -> List['TwoDQubit']:
        """Returns a square of TwoDQubit.

        Args:
            diameter: Length of a side of the square.
            x0: x-coordinate of the first qubit.
            y0: y-coordinate of the first qubit.

        Returns:
            A list of TwoDQubits filling in a square grid
        """
        return TwoDQubit.rect(diameter, diameter, x0=x0, y0=y0)

    @staticmethod
    def rect(rows: int, cols: int, x0: float=0, y0: float=0) -> List['TwoDQubit']:
        """Returns a rectangle of TwoDQubit.

        Args:
            rows: Number of rows in the rectangle.
            cols: Number of columns in the rectangle.
            x0: x-coordinate of the first qubit.
            y0: y-coordinate of the first qubit.

        Returns:
            A list of TwoDQubits filling in a rectangular grid
        """
        return [TwoDQubit(x0 + x, y0 + y) for y in range(cols) for x in range(rows)]

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

    def __repr__(self):
        return f'pasqal.TwoDQubit({self.x}, {self.y})'

    def __str__(self):
        return f'({self.x}, {self.y})'

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['x', 'y'])