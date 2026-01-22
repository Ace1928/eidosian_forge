from typing import Callable, Iterator, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, value
from cirq.interop.quirk.cells.cell import CELL_SIZES, CellMaker
def _deinterleave_bit(n: int, x: int) -> int:
    h = (n + 1) // 2
    stride = x // 2
    group = x % 2
    return stride + group * h