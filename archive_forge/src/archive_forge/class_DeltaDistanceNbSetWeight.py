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
class DeltaDistanceNbSetWeight(NbSetWeight):
    """Weight of neighbors set based on the difference of distances."""
    SHORT_NAME = 'DeltaDistanceNbSetWeight'

    def __init__(self, weight_function=None, nbs_source='voronoi'):
        """Initialize DeltaDistanceNbSetWeight.

        Args:
            weight_function: Ratio function to use.
            nbs_source: Source of the neighbors.
        """
        if weight_function is None:
            self.weight_function = {'function': 'smootherstep', 'options': {'lower': 0.1, 'upper': 0.2}}
        else:
            self.weight_function = weight_function
        self.weight_rf = RatioFunction.from_dict(self.weight_function)
        if nbs_source not in ['nb_sets', 'voronoi']:
            raise ValueError('"nbs_source" should be one of ["nb_sets", "voronoi"]')
        self.nbs_source = nbs_source

    def weight(self, nb_set, structure_environments, cn_map=None, additional_info=None):
        """Get the weight of a given neighbors set.

        Args:
            nb_set: Neighbors set.
            structure_environments: Structure environments used to estimate weight.
            cn_map: Mapping index for this neighbors set.
            additional_info: Additional information.

        Returns:
            Weight of the neighbors set.
        """
        cn = cn_map[0]
        isite = nb_set.isite
        voronoi = structure_environments.voronoi.voronoi_list2[isite]
        if self.nbs_source == 'nb_sets':
            all_nbs_voro_indices = set()
            for cn2, nb_sets in structure_environments.neighbors_sets[isite].items():
                for nb_set2 in nb_sets:
                    if cn == cn2:
                        continue
                    all_nbs_voro_indices.update(nb_set2.site_voronoi_indices)
        elif self.nbs_source == 'voronoi':
            all_nbs_voro_indices = set(range(len(voronoi)))
        else:
            raise ValueError('"nbs_source" should be one of ["nb_sets", "voronoi"]')
        all_nbs_indices_except_nb_set = all_nbs_voro_indices - nb_set.site_voronoi_indices
        normalized_distances = [voronoi[inb]['normalized_distance'] for inb in all_nbs_indices_except_nb_set]
        if len(normalized_distances) == 0:
            return 1
        if len(nb_set) == 0:
            return 0
        nb_set_max_normalized_distance = max(nb_set.normalized_distances)
        return self.weight_rf.eval(min(normalized_distances) - nb_set_max_normalized_distance)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def as_dict(self):
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'weight_function': self.weight_function, 'nbs_source': self.nbs_source}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize from dict.

        Args:
            dct (dict): Dict representation of DeltaDistanceNbSetWeight.

        Returns:
            DeltaDistanceNbSetWeight.
        """
        return cls(weight_function=dct['weight_function'], nbs_source=dct['nbs_source'])