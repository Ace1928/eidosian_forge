from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@vacuum_over_film.setter
def vacuum_over_film(self, new_vacuum: float) -> None:
    if new_vacuum < 0:
        raise ValueError('The vacuum over the film can not be less then 0')
    delta = new_vacuum - self.vacuum_over_film
    self._vacuum_over_film = new_vacuum
    self._update_c(self.lattice.c + delta)