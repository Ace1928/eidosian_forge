from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@classmethod
def type_description(cls):
    """Return complete description of this Bravais lattice type."""
    desc = 'Lattice name: {name}\n  Long name: {longname}\n  Parameters: {parameters}\n'.format(**vars(cls))
    chunks = [desc]
    for name in cls.variant_names:
        var = cls.variants[name]
        txt = str(var)
        lines = ['  ' + L for L in txt.splitlines()]
        lines.append('')
        chunks.extend(lines)
    return '\n'.join(chunks)