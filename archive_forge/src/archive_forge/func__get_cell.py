import re
import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from .parser import _define_pattern
def _get_cell(text):
    cell = np.zeros((3, 3))
    lattice = _cell_3d.findall(text)
    if lattice:
        pbc = [True, True, True]
        for i, row in enumerate(lattice[0].strip().split('\n')):
            cell[i] = [float(x) for x in row.split()]
        return (cell, pbc)
    pbc = [False, False, False]
    lengths = [None, None, None]
    angles = [None, None, None]
    for row in text.strip().split('\n'):
        row = row.strip().lower()
        for dim, vecname in enumerate(['a', 'b', 'c']):
            if row.startswith('lat_{}'.format(vecname)):
                pbc[dim] = True
                lengths[dim] = float(row.split()[1])
        for i, angle in enumerate(['alpha', 'beta', 'gamma']):
            if row.startswith(angle):
                angles[i] = float(row.split()[1])
    if not np.any(pbc):
        return (None, pbc)
    for i in range(3):
        a, b, c = np.roll(np.array([0, 1, 2]), i)
        if pbc[a] and pbc[b]:
            assert angles[c] is not None
        if angles[c] is not None:
            assert pbc[a] and pbc[b]
    if np.all(pbc):
        return (cellpar_to_cell(lengths + angles), pbc)
    if np.sum(pbc) == 1:
        dim = np.argmax(pbc)
        cell[dim, dim] = lengths[dim]
        return (cell, pbc)
    dim1, dim2 = [dim for dim, ipbc in enumerate(pbc) if ipbc]
    angledim = np.argmin(pbc)
    cell[dim1, dim1] = lengths[dim1]
    cell[dim2, dim2] = lengths[dim2] * np.sin(angles[angledim])
    cell[dim2, dim1] = lengths[dim2] * np.cos(angles[angledim])
    return (cell, pbc)