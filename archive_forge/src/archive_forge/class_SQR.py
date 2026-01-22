from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive square', 'tetragonal', None, 'tp', ('a',), [['SQR', 'GMX', 'MGXM', get_subset_points('GMX', sc_special_points['tetragonal'])]], ndim=2)
class SQR(BravaisLattice):

    def __init__(self, a, **kwargs):
        BravaisLattice.__init__(self, a=a, **kwargs)

    def _cell(self, a):
        return np.array([[a, 0, 0], [0, a, 0], [0, 0, 0.0]])