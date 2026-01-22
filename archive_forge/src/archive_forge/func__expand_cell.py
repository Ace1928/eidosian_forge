from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def _expand_cell(cell: Cell) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    lattice = np.array(np.transpose(cell[0]), dtype='double', order='C')
    positions = np.array(cell[1], dtype='double', order='C')
    numbers = np.array(cell[2], dtype='intc')
    if len(cell) == 4:
        magmoms = np.array(cell[3], order='C', dtype='double')
    elif len(cell) == 3:
        magmoms = None
    else:
        raise TypeError('cell has to be a tuple of 3 or 4 elements.')
    if lattice.shape != (3, 3):
        raise TypeError('lattice has to be a (3, 3) array.')
    if not (positions.ndim == 2 and positions.shape[1] == 3):
        raise TypeError('positions has to be a (num_atoms, 3) array.')
    num_atoms = positions.shape[0]
    if numbers.ndim != 1:
        raise TypeError('numbers has to be a (num_atoms,) array.')
    if len(numbers) != num_atoms:
        raise TypeError('numbers has to have the same number of atoms as positions.')
    if magmoms is not None:
        if len(magmoms) != num_atoms:
            raise TypeError('magmoms has to have the same number of atoms as positions.')
        if magmoms.ndim == 1:
            pass
        elif magmoms.ndim == 2:
            if magmoms.shape[1] != 3:
                raise TypeError('non-collinear magmoms has to be a (num_atoms, 3) array.')
        else:
            raise TypeError('magmoms has to be a 1D or 2D array.')
    return (lattice, positions, numbers, magmoms)