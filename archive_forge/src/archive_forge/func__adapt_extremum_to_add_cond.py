from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
def _adapt_extremum_to_add_cond(self, list_icohps, percentage):
    """
        Convinicence method for returning the extremum of the given icohps or icoops or icobis list

        Args:
            list_icohps: can be a list of icohps or icobis or icobis

        Returns:
            float: min value of input list of icohps / max value of input list of icobis or icobis
        """
    which_extr = min if not self.are_coops and (not self.are_cobis) else max
    return which_extr(list_icohps) * percentage