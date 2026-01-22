import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Union
from typing_extensions import Self
from cirq import ops, protocols
class LineQid(_BaseLineQid):
    """A qid on a 1d lattice with nearest-neighbor connectivity.

    `LineQid`s have a single attribute, and integer coordinate 'x', which
    identifies the qids location on the line. `LineQid`s are ordered by
    this integer.

    One can construct new `cirq.LineQid`s by adding or subtracting integers:

    >>> cirq.LineQid(1, dimension=2) + 3
    cirq.LineQid(4, dimension=2)

    >>> cirq.LineQid(2, dimension=3) - 1
    cirq.LineQid(1, dimension=3)

    """

    def __init__(self, x: int, dimension: int) -> None:
        """Initializes a line qid at the given x coordinate.

        Args:
            x: The x coordinate.
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.
        """
        self.validate_dimension(dimension)
        self._x = x
        self._dimension = dimension

    def _with_x(self, x: int) -> 'LineQid':
        return LineQid(x, dimension=self._dimension)

    @staticmethod
    def range(*range_args, dimension: int) -> List['LineQid']:
        """Returns a range of line qids.

        Args:
            *range_args: Same arguments as python's built-in range method.
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.

        Returns:
            A list of line qids.
        """
        return [LineQid(i, dimension=dimension) for i in range(*range_args)]

    @staticmethod
    def for_qid_shape(qid_shape: Sequence[int], start: int=0, step: int=1) -> List['LineQid']:
        """Returns a range of line qids for each entry in `qid_shape` with
        matching dimension.

        Args:
            qid_shape: A sequence of dimensions for each `LineQid` to create.
            start: The x coordinate of the first `LineQid`.
            step: The amount to increment each x coordinate.
        """
        return [LineQid(start + step * i, dimension=dimension) for i, dimension in enumerate(qid_shape)]

    @staticmethod
    def for_gate(val: Any, start: int=0, step: int=1) -> List['LineQid']:
        """Returns a range of line qids with the same qid shape as the gate.

        Args:
            val: Any value that supports the `cirq.qid_shape` protocol.  Usually
                a gate.
            start: The x coordinate of the first `LineQid`.
            step: The amount to increment each x coordinate.
        """
        from cirq.protocols.qid_shape_protocol import qid_shape
        return LineQid.for_qid_shape(qid_shape(val), start=start, step=step)

    def __repr__(self) -> str:
        return f'cirq.LineQid({self._x}, dimension={self._dimension})'

    def __str__(self) -> str:
        return f'q({self._x}) (d={self._dimension})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=(f'{self._x} (d={self._dimension})',))

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['x', 'dimension'])