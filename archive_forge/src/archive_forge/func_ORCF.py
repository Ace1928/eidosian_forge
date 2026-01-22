from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def ORCF(self):
    prods = self.prods
    if all(prods[3:] > 0):
        orcf_abc = 2 * np.sqrt(prods[3:])
        return self._check(ORCF, *orcf_abc)