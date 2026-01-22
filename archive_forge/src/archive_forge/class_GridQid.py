import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, TYPE_CHECKING, Union
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols
class GridQid(_BaseGridQid):
    """A qid on a 2d square lattice

    GridQid uses row-major ordering:

        GridQid(0, 0, dimension=2) < GridQid(0, 1, dimension=2)
        < GridQid(1, 0, dimension=2) < GridQid(1, 1, dimension=2)

    New GridQid can be constructed by adding or subtracting tuples or numpy
    arrays

    >>> cirq.GridQid(2, 3, dimension=2) + (3, 1)
    cirq.GridQid(5, 4, dimension=2)
    >>> cirq.GridQid(2, 3, dimension=2) - (1, 2)
    cirq.GridQid(1, 1, dimension=2)
    >>> cirq.GridQid(2, 3, dimension=2) + np.array([3, 1], dtype=int)
    cirq.GridQid(5, 4, dimension=2)
    """

    def __init__(self, row: int, col: int, *, dimension: int) -> None:
        """Initializes a grid qid at the given row, col coordinate

        Args:
            row: the row coordinate
            col: the column coordinate
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.
        """
        self.validate_dimension(dimension)
        self._row = row
        self._col = col
        self._dimension = dimension

    def _with_row_col(self, row: int, col: int) -> 'GridQid':
        return GridQid(row, col, dimension=self.dimension)

    @staticmethod
    def square(diameter: int, top: int=0, left: int=0, *, dimension: int) -> List['GridQid']:
        """Returns a square of GridQid.

        Args:
            diameter: Length of a side of the square
            top: Row number of the topmost row
            left: Column number of the leftmost row
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.

        Returns:
            A list of GridQid filling in a square grid
        """
        return GridQid.rect(diameter, diameter, top=top, left=left, dimension=dimension)

    @staticmethod
    def rect(rows: int, cols: int, top: int=0, left: int=0, *, dimension: int) -> List['GridQid']:
        """Returns a rectangle of GridQid.

        Args:
            rows: Number of rows in the rectangle
            cols: Number of columns in the rectangle
            top: Row number of the topmost row
            left: Column number of the leftmost row
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.

        Returns:
            A list of GridQid filling in a rectangular grid
        """
        return [GridQid(row, col, dimension=dimension) for row in range(top, top + rows) for col in range(left, left + cols)]

    @staticmethod
    def from_diagram(diagram: str, dimension: int) -> List['GridQid']:
        """Parse ASCII art device layout into a device.

        As an example, the below diagram will create a list of GridQid in a
        pyramid structure.


        ```
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA
        ```

        You can use any character other than a hyphen, period or space to mark a
        qid. As an example, the qids for a Bristlecone device could be
        represented by the below diagram. This produces a diamond-shaped grid of
        qids, and qids with the same letter correspond to the same readout line.

        ```
        .....AB.....
        ....ABCD....
        ...ABCDEF...
        ..ABCDEFGH..
        .ABCDEFGHIJ.
        ABCDEFGHIJKL
        .CDEFGHIJKL.
        ..EFGHIJKL..
        ...GHIJKL...
        ....IJKL....
        .....KL.....
        ```

        Args:
            diagram: String representing the qid layout. Each line represents
                a row. Alphanumeric characters are assigned as qid.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('\\n'), an error will
                be thrown. The top-left corner of the diagram will be have
                coordinate (0, 0).

            dimension: The dimension of the qubits in the `cirq.GridQid`s used
                in this construction.

        Returns:
            A list of `cirq.GridQid`s corresponding to qids in the provided diagram

        Raises:
            ValueError: If the input string contains an invalid character.
        """
        coords = _ascii_diagram_to_coords(diagram)
        return [GridQid(*c, dimension=dimension) for c in coords]

    def __repr__(self) -> str:
        return f'cirq.GridQid({self._row}, {self._col}, dimension={self.dimension})'

    def __str__(self) -> str:
        return f'q({self._row}, {self._col}) (d={self.dimension})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=(f'({self._row}, {self._col}) (d={self.dimension})',))

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['row', 'col', 'dimension'])