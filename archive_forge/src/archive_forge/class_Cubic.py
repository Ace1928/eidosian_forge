from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
class Cubic(BravaisLattice):
    """Abstract class for cubic lattices."""
    conventional_cls = 'CUB'

    def __init__(self, a):
        BravaisLattice.__init__(self, a=a)