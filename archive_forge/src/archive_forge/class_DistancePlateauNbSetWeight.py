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
class DistancePlateauNbSetWeight(NbSetWeight):
    """Weight of neighbors set based on the distance."""
    SHORT_NAME = 'DistancePlateauWeight'

    def __init__(self, distance_function=None, weight_function=None):
        """Initialize DistancePlateauNbSetWeight.

        Args:
            distance_function: Distance function to use.
            weight_function: Ratio function to use.
        """
        if distance_function is None:
            self.distance_function = {'type': 'normalized_distance'}
        else:
            self.distance_function = distance_function
        if weight_function is None:
            self.weight_function = {'function': 'inverse_smootherstep', 'options': {'lower': 0.2, 'upper': 0.4}}
        else:
            self.weight_function = weight_function
        self.weight_rf = RatioFunction.from_dict(self.weight_function)

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
        return self.weight_rf.eval(nb_set.distance_plateau())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def as_dict(self):
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'distance_function': self.distance_function, 'weight_function': self.weight_function}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize from dict.

        Args:
            dct (dict): Dict representation of DistancePlateauNbSetWeight.

        Returns:
            DistancePlateauNbSetWeight.
        """
        return cls(distance_function=dct['distance_function'], weight_function=dct['weight_function'])