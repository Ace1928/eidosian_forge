from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from monty.json import MSONable
from scipy.stats import gmean
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import EquivalentSiteSearchError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import get_lower_and_upper_f
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.func_utils import (
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def w_area_has_intersection(self, nb_set, structure_environments, cn_map, additional_info):
    """Get intersection of the neighbors set area with the surface.

        Args:
            nb_set: Neighbors set.
            structure_environments: Structure environments.
            cn_map: Mapping index of the neighbors set.
            additional_info: Additional information.

        Returns:
            Area intersection between neighbors set and surface.
        """
    return self.w_area_intersection_specific(nb_set=nb_set, structure_environments=structure_environments, cn_map=cn_map, additional_info=additional_info)