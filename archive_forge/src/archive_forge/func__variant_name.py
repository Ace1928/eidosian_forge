from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
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