import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
def _mod_inv_else_1(a: int, m: int) -> int:
    """Returns `a**-1` if `a` has a multiplicative inverse, else 1."""
    if m == 0:
        return 1
    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        return 1
    return x % m