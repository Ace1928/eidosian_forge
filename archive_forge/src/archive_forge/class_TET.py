from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive tetragonal', 'tetragonal', 'tetragonal', 'tP', 'ac', [['TET', 'GAMRXZ', 'GXMGZRAZ,XR,MA', sc_special_points['tetragonal']]])
class TET(BravaisLattice):
    conventional_cls = 'TET'
    conventional_cellmap = _identity

    def __init__(self, a, c):
        BravaisLattice.__init__(self, a=a, c=c)

    def _cell(self, a, c):
        return np.diag(np.array([a, a, c]))