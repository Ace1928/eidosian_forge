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
def _parse_lattice(b):
    """Parses compact binary representation into list of lattice vectors/centerings."""
    if len(b) == 0:
        return None
    raw_lattice = [b[i:i + 4] for i in range(0, len(b), 4)]
    lattice = []
    for r in raw_lattice:
        lattice.append({'vector': [r[0] / r[3], r[1] / r[3], r[2] / r[3]], 'str': f'({Fraction(r[0] / r[3]).limit_denominator()},{Fraction(r[1] / r[3]).limit_denominator()},{Fraction(r[2] / r[3]).limit_denominator()})+'})
    return lattice