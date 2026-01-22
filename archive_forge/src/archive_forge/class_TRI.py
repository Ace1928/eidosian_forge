from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive triclinic', 'triclinic', 'triclinic', 'aP', ('a', 'b', 'c', 'alpha', 'beta', 'gamma'), [['TRI1a', 'GLMNRXYZ', 'XGY,LGZ,NGM,RG', None], ['TRI2a', 'GLMNRXYZ', 'XGY,LGZ,NGM,RG', None], ['TRI1b', 'GLMNRXYZ', 'XGY,LGZ,NGM,RG', None], ['TRI2b', 'GLMNRXYZ', 'XGY,LGZ,NGM,RG', None]])
class TRI(BravaisLattice):
    conventional_cls = 'TRI'
    conventional_cellmap = _identity

    def __init__(self, a, b, c, alpha, beta, gamma):
        BravaisLattice.__init__(self, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

    def _cell(self, a, b, c, alpha, beta, gamma):
        alpha, beta, gamma = np.array([alpha, beta, gamma])
        singamma = np.sin(gamma * _degrees)
        cosgamma = np.cos(gamma * _degrees)
        cosbeta = np.cos(beta * _degrees)
        cosalpha = np.cos(alpha * _degrees)
        a3x = c * cosbeta
        a3y = c / singamma * (cosalpha - cosbeta * cosgamma)
        a3z = c / singamma * np.sqrt(singamma ** 2 - cosalpha ** 2 - cosbeta ** 2 + 2 * cosalpha * cosbeta * cosgamma)
        return np.array([[a, 0, 0], [b * cosgamma, b * singamma, 0], [a3x, a3y, a3z]])

    def _variant_name(self, a, b, c, alpha, beta, gamma):
        cell = Cell.new([a, b, c, alpha, beta, gamma])
        icellpar = Cell(cell.reciprocal()).cellpar()
        kangles = kalpha, kbeta, kgamma = icellpar[3:]

        def raise_unconventional():
            raise UnconventionalLattice(tri_angles_explanation.format(*kangles))
        eps = self._eps
        if abs(kgamma - 90) < eps:
            if kalpha > 90 and kbeta > 90:
                var = '2a'
            elif kalpha < 90 and kbeta < 90:
                var = '2b'
            else:
                raise_unconventional()
        elif all(kangles > 90):
            if kgamma > min(kangles):
                raise_unconventional()
            var = '1a'
        elif all(kangles < 90):
            if kgamma < max(kangles):
                raise_unconventional()
            var = '1b'
        else:
            raise_unconventional()
        return 'TRI' + var

    def _special_points(self, a, b, c, alpha, beta, gamma, variant):
        if variant.name == 'TRI1a' or variant.name == 'TRI2a':
            points = [[0.0, 0.0, 0.0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
        else:
            points = [[0, 0, 0], [0.5, -0.5, 0], [0, 0, 0.5], [-0.5, -0.5, 0.5], [0, -0.5, 0.5], [0, -0.5, 0], [0.5, 0, 0], [-0.5, 0, 0.5]]
        return points