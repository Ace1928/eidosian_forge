from __future__ import annotations
import os
import sqlite3
import textwrap
from array import array
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from monty.design_patterns import cached_class
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.groups import SymmetryGroup, in_array_list
from pymatgen.symmetry.settings import JonesFaithfulTransformation
from pymatgen.util.string import transformation_to_string
def _parse_wyckoff(b):
    """Parses compact binary representation into list of Wyckoff sites."""
    if len(b) == 0:
        return None
    wyckoff_sites = []

    def get_label(idx):
        if idx <= 25:
            return chr(97 + idx)
        return 'alpha'
    o = 0
    n = 1
    num_wyckoff = b[0]
    while len(wyckoff_sites) < num_wyckoff:
        m = b[1 + o]
        label = str(b[2 + o] * m) + get_label(num_wyckoff - n)
        sites = []
        for j in range(m):
            s = b[3 + o + j * 22:3 + o + j * 22 + 22]
            translation_vec = [s[0] / s[3], s[1] / s[3], s[2] / s[3]]
            matrix = [[s[4], s[7], s[10]], [s[5], s[8], s[11]], [s[6], s[9], s[12]]]
            matrix_magmom = [[s[13], s[16], s[19]], [s[14], s[17], s[20]], [s[15], s[18], s[21]]]
            wyckoff_str = f'({transformation_to_string(matrix, translation_vec)};{transformation_to_string(matrix_magmom, c='m')})'
            sites.append({'translation_vec': translation_vec, 'matrix': matrix, 'matrix_magnetic': matrix_magmom, 'str': wyckoff_str})
        wyckoff_sites.append({'label': label, 'str': ' '.join((s['str'] for s in sites))})
        n += 1
        o += m * 22 + 2
    return wyckoff_sites