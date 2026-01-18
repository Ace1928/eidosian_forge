import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def translate_pretty(fractional, pbc):
    """Translates atoms such that fractional positions are minimized."""
    for i in range(3):
        if not pbc[i]:
            continue
        indices = np.argsort(fractional[:, i])
        sp = fractional[indices, i]
        widths = (np.roll(sp, 1) - sp) % 1.0
        fractional[:, i] -= sp[np.argmin(widths)]
        fractional[:, i] %= 1.0
    return fractional