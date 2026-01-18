from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
def update_coordination_environments(self, isite, cn, nb_set, ce):
    """
        Updates the coordination environment for this site, coordination and neighbor set.

        Args:
            isite: Index of the site to be updated.
            cn: Coordination to be updated.
            nb_set: Neighbors set to be updated.
            ce: ChemicalEnvironments object for this neighbors set.
        """
    if self.ce_list[isite] is None:
        self.ce_list[isite] = {}
    if cn not in self.ce_list[isite]:
        self.ce_list[isite][cn] = []
    try:
        nb_set_index = self.neighbors_sets[isite][cn].index(nb_set)
    except ValueError:
        raise ValueError('Neighbors set not found in the structure environments')
    if nb_set_index == len(self.ce_list[isite][cn]):
        self.ce_list[isite][cn].append(ce)
    elif nb_set_index < len(self.ce_list[isite][cn]):
        self.ce_list[isite][cn][nb_set_index] = ce
    else:
        raise ValueError('Neighbors set not yet in ce_list !')