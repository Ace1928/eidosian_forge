import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def _read_cell(data):
    """Helper to read cell data, returns cell and pbc"""
    cell = None
    pbc = [False, False, False]
    if 'CELL' in data:
        content = data['CELL']['content']
        cell = Cell([[0, 0, 0] for i in range(3)])
        char2idx = {'A ': 0, 'B ': 1, 'C ': 2}
        for line in content:
            if line[:2] in char2idx:
                idx = char2idx[line[:2]]
                cell[idx] = [float(x) for x in line.split()[1:]]
                pbc[idx] = True
        if not set([len(v) for v in cell]) == {3}:
            raise RuntimeError('Bad Cell Definition found.')
    return (cell, pbc)