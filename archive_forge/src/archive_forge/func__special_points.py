from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def _special_points(self, a, alpha, variant):
    if alpha > 90:
        _alpha = 180 - alpha
    else:
        _alpha = alpha
    sina2 = np.sin(_alpha / 2 * _degrees) ** 2
    sina = np.sin(_alpha * _degrees) ** 2
    eta = sina2 / sina
    cosa = np.cos(_alpha * _degrees)
    xi = eta * cosa
    points = [[0, 0, 0], [eta, -eta, 0], [0.5 + xi, 0.5 - xi, 0], [0.5, 0.5, 0]]
    if alpha > 90:
        op = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        points = np.dot(points, op.T)
    return points