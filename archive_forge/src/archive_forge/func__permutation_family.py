from typing import Callable, Iterator, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, value
from cirq.interop.quirk.cells.cell import CELL_SIZES, CellMaker
def _permutation_family(identifier_prefix: str, name: str, permute: Callable[[int, int], int]) -> Iterator[CellMaker]:
    for n in CELL_SIZES:
        permutation = tuple((permute(n, i) % n for i in range(n)))
        yield _permutation(identifier_prefix + str(n), name, permutation)