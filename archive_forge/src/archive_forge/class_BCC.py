from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('body-centred cubic', 'cubic', 'cubic', 'cI', 'a', [['BCC', 'GHPN', 'GHNGPH,PN', sc_special_points['bcc']]])
class BCC(Cubic):
    conventional_cellmap = _fcc_map

    def _cell(self, a):
        return 0.5 * np.array([[-a, a, a], [a, -a, a], [a, a, -a]])