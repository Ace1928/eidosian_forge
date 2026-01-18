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
def w_area_intersection_nbsfh_fbs_onb0(self, nb_set, structure_environments, cn_map, additional_info):
    """Get intersection of the neighbors set area with the surface.

        Args:
            nb_set: Neighbors set.
            structure_environments: Structure environments.
            cn_map: Mapping index of the neighbors set.
            additional_info: Additional information.

        Returns:
            Area intersection between neighbors set and surface.
        """
    dist_ang_sources = [src for src in nb_set.sources if src['origin'] == 'dist_ang_ac_voronoi' and src['ac'] == self.additional_condition]
    if len(dist_ang_sources) > 0:
        for src in dist_ang_sources:
            d1 = src['dp_dict']['min']
            d2 = src['dp_dict']['next']
            a1 = src['ap_dict']['next']
            a2 = src['ap_dict']['max']
            if self.rectangle_crosses_area(d1=d1, d2=d2, a1=a1, a2=a2):
                return 1
        return 0
    from_hints_sources = [src for src in nb_set.sources if src['origin'] == 'nb_set_hints']
    if len(from_hints_sources) == 0:
        return 0
    if len(from_hints_sources) != 1:
        raise ValueError('Found multiple hints sources for nb_set')
    cn_map_src = from_hints_sources[0]['cn_map_source']
    nb_set_src = structure_environments.neighbors_sets[nb_set.isite][cn_map_src[0]][cn_map_src[1]]
    dist_ang_sources = [src for src in nb_set_src.sources if src['origin'] == 'dist_ang_ac_voronoi' and src['ac'] == self.additional_condition]
    if len(dist_ang_sources) == 0:
        return 0
    for src in dist_ang_sources:
        d1 = src['dp_dict']['min']
        d2 = src['dp_dict']['next']
        a1 = src['ap_dict']['next']
        a2 = src['ap_dict']['max']
        if self.rectangle_crosses_area(d1=d1, d2=d2, a1=a1, a2=a2):
            return 1
    return 0