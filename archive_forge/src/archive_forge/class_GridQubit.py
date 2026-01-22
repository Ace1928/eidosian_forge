import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, TYPE_CHECKING, Union
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols
class GridQubit(_BaseGridQid):
    """A qubit on a 2d square lattice.

    GridQubits use row-major ordering:

        GridQubit(0, 0) < GridQubit(0, 1) < GridQubit(1, 0) < GridQubit(1, 1)

    New GridQubits can be constructed by adding or subtracting tuples

    >>> cirq.GridQubit(2, 3) + (3, 1)
    cirq.GridQubit(5, 4)
    >>> cirq.GridQubit(2, 3) - (1, 2)
    cirq.GridQubit(1, 1)
    >>> cirq.GridQubit(2, 3,) + np.array([3, 1], dtype=int)
    cirq.GridQubit(5, 4)
    """
    _dimension = 2

    def __init__(self, row: int, col: int) -> None:
        self._row = row
        self._col = col

    def _with_row_col(self, row: int, col: int):
        return GridQubit(row, col)

    def _cmp_tuple(self):
        cls = GridQid if type(self) is GridQubit else type(self)
        return (cls.__name__, repr(cls), self._comparison_key(), self.dimension)

    @staticmethod
    def square(diameter: int, top: int=0, left: int=0) -> List['GridQubit']:
        """Returns a square of GridQubits.

        Args:
            diameter: Length of a side of the square
            top: Row number of the topmost row
            left: Column number of the leftmost row

        Returns:
            A list of GridQubits filling in a square grid
        """
        return GridQubit.rect(diameter, diameter, top=top, left=left)

    @staticmethod
    def rect(rows: int, cols: int, top: int=0, left: int=0) -> List['GridQubit']:
        """Returns a rectangle of GridQubits.

        Args:
            rows: Number of rows in the rectangle
            cols: Number of columns in the rectangle
            top: Row number of the topmost row
            left: Column number of the leftmost row

        Returns:
            A list of GridQubits filling in a rectangular grid
        """
        return [GridQubit(row, col) for row in range(top, top + rows) for col in range(left, left + cols)]

    @staticmethod
    def from_diagram(diagram: str) -> List['GridQubit']:
        """Parse ASCII art into device layout info.

        As an example, the below diagram will create a list of
        GridQubit in a pyramid structure.

        ```
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA
        ```

        You can use any character other than a hyphen, period or space to mark
        a qubit. As an example, the qubits for a Bristlecone device could be
        represented by the below diagram. This produces a diamond-shaped grid
        of qids, and qids with the same letter correspond to the same readout
        line.

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
            diagram: String representing the qubit layout. Each line represents
                a row. Alphanumeric characters are assigned as qid.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('\\n'), an error will
                be thrown. The top-left corner of the diagram will be have
                coordinate (0,0).

        Returns:
            A list of GridQubit corresponding to qubits in the provided diagram

        Raises:
            ValueError: If the input string contains an invalid character.
        """
        coords = _ascii_diagram_to_coords(diagram)
        return [GridQubit(*c) for c in coords]

    def __repr__(self) -> str:
        return f'cirq.GridQubit({self._row}, {self._col})'

    def __str__(self) -> str:
        return f'q({self._row}, {self._col})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=(f'({self._row}, {self._col})',))

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['row', 'col'])