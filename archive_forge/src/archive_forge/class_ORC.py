from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive orthorhombic', 'orthorhombic', 'orthorhombic', 'oP', 'abc', [['ORC', 'GRSTUXYZ', 'GXSYGZURTZ,YT,UX,SR', sc_special_points['orthorhombic']]])
class ORC(Orthorhombic):
    conventional_cls = 'ORC'
    conventional_cellmap = _identity

    def _cell(self, a, b, c):
        return np.diag([a, b, c]).astype(float)