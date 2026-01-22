from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('base-centred orthorhombic', 'orthorhombic', 'orthorhombic', 'oC', 'abc', [['ORCC', 'GAA1RSTXX1YZ', 'GXSRAZGYX1A1TY,ZT', None]])
class ORCC(BravaisLattice):
    conventional_cls = 'ORC'
    conventional_cellmap = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])

    def __init__(self, a, b, c):
        if a >= b:
            raise UnconventionalLattice(f'Expected a < b, got a={a}, b={b}')
        BravaisLattice.__init__(self, a=a, b=b, c=c)

    def _cell(self, a, b, c):
        return np.array([[0.5 * a, -0.5 * b, 0], [0.5 * a, 0.5 * b, 0], [0, 0, c]])

    def _special_points(self, a, b, c, variant):
        zeta = 0.25 * (1 + a * a / (b * b))
        points = [[0, 0, 0], [zeta, zeta, 0.5], [-zeta, 1 - zeta, 0.5], [0, 0.5, 0.5], [0, 0.5, 0], [-0.5, 0.5, 0.5], [zeta, zeta, 0], [-zeta, 1 - zeta, 0], [-0.5, 0.5, 0], [0, 0, 0.5]]
        return points