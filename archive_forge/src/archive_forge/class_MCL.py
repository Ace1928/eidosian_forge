from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive monoclinic', 'monoclinic', 'monoclinic', 'mP', ('a', 'b', 'c', 'alpha'), [['MCL', 'GACDD1EHH1H2MM1M2XYY1Z', 'GYHCEM1AXH1,MDZ,YD', None]])
class MCL(BravaisLattice):
    conventional_cls = 'MCL'
    conventional_cellmap = _identity

    def __init__(self, a, b, c, alpha):
        check_mcl(a, b, c, alpha)
        BravaisLattice.__init__(self, a=a, b=b, c=c, alpha=alpha)

    def _cell(self, a, b, c, alpha):
        alpha *= _degrees
        return np.array([[a, 0, 0], [0, b, 0], [0, c * np.cos(alpha), c * np.sin(alpha)]])

    def _special_points(self, a, b, c, alpha, variant):
        cosa = np.cos(alpha * _degrees)
        eta = (1 - b * cosa / c) / (2 * np.sin(alpha * _degrees) ** 2)
        nu = 0.5 - eta * c * cosa / b
        points = [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0, -0.5], [0.5, 0.5, 0.5], [0, eta, 1 - nu], [0, 1 - eta, nu], [0, eta, -nu], [0.5, eta, 1 - nu], [0.5, 1 - eta, nu], [0.5, eta, -nu], [0, 0.5, 0], [0, 0, 0.5], [0, 0, -0.5], [0.5, 0, 0]]
        return points

    def _variant_name(self, a, b, c, alpha):
        check_mcl(a, b, c, alpha)
        return 'MCL'