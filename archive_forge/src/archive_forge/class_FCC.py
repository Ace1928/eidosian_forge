from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('face-centred cubic', 'cubic', 'cubic', 'cF', 'a', [['FCC', 'GKLUWX', 'GXWKGLUWLK,UX', sc_special_points['fcc']]])
class FCC(Cubic):
    conventional_cellmap = _bcc_map

    def _cell(self, a):
        return 0.5 * np.array([[0.0, a, a], [a, 0, a], [a, a, 0]])