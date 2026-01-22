import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def _read_geometry(content):
    """Helper to read geometry, returns a list of Atoms"""
    atom_list = []
    for entry in content:
        entry = entry.split()
        el = [char.lower() for char in entry[0] if char.isalpha()]
        el = ''.join(el).capitalize()
        pos = [float(x) for x in entry[1:4]]
        if el in atomic_numbers.keys():
            atom_list.append(Atom(el, pos))
        else:
            atom_list.append(Atom('X', pos))
    return atom_list