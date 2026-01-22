from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive cubic', 'cubic', 'cubic', 'cP', 'a', [['CUB', 'GXRM', 'GXMGRX,MR', sc_special_points['cubic']]])
class CUB(Cubic):
    conventional_cellmap = _identity

    def _cell(self, a):
        return a * np.eye(3)