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
class CNBiasNbSetWeight(NbSetWeight):
    """Weight of neighbors set based on specific biases towards specific coordination numbers."""
    SHORT_NAME = 'CNBiasWeight'

    def __init__(self, cn_weights, initialization_options):
        """Initialize CNBiasNbSetWeight.

        Args:
            cn_weights: Weights for each coordination.
            initialization_options: Options for initialization.
        """
        self.cn_weights = cn_weights
        self.initialization_options = initialization_options

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
        return self.cn_weights[len(nb_set)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NbSetWeight):
            return NotImplemented
        return self.cn_weights == other.cn_weights and self.initialization_options == other.initialization_options

    def as_dict(self):
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'cn_weights': {str(cn): cnw for cn, cnw in self.cn_weights.items()}, 'initialization_options': self.initialization_options}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize from dict.

        Args:
            dct (dict): Dict representation of CNBiasNbSetWeight.

        Returns:
            CNBiasNbSetWeight.
        """
        return cls(cn_weights={int(cn): cnw for cn, cnw in dct['cn_weights'].items()}, initialization_options=dct['initialization_options'])

    @classmethod
    def linearly_equidistant(cls, weight_cn1, weight_cn13):
        """Initialize linearly equidistant weights for each coordination.

        Args:
            weight_cn1: Weight of coordination 1.
            weight_cn13: Weight of coordination 13.

        Returns:
            CNBiasNbSetWeight.
        """
        initialization_options = {'type': 'linearly_equidistant', 'weight_cn1': weight_cn1, 'weight_cn13': weight_cn13}
        dw = (weight_cn13 - weight_cn1) / 12.0
        cn_weights = {cn: weight_cn1 + (cn - 1) * dw for cn in range(1, 14)}
        return cls(cn_weights=cn_weights, initialization_options=initialization_options)

    @classmethod
    def geometrically_equidistant(cls, weight_cn1, weight_cn13):
        """Initialize geometrically equidistant weights for each coordination.

        Arge:
            weight_cn1: Weight of coordination 1.
            weight_cn13: Weight of coordination 13.

        Returns:
            CNBiasNbSetWeight.
        """
        initialization_options = {'type': 'geometrically_equidistant', 'weight_cn1': weight_cn1, 'weight_cn13': weight_cn13}
        factor = np.power(float(weight_cn13) / weight_cn1, 1 / 12.0)
        cn_weights = {cn: weight_cn1 * np.power(factor, cn - 1) for cn in range(1, 14)}
        return cls(cn_weights=cn_weights, initialization_options=initialization_options)

    @classmethod
    def explicit(cls, cn_weights):
        """Initialize weights explicitly for each coordination.

        Args:
            cn_weights: Weights for each coordination.

        Returns:
            CNBiasNbSetWeight.
        """
        initialization_options = {'type': 'explicit'}
        if set(cn_weights) != set(range(1, 14)):
            raise ValueError('Weights should be provided for CN 1 to 13')
        return cls(cn_weights=cn_weights, initialization_options=initialization_options)

    @classmethod
    def from_description(cls, dct: dict) -> Self:
        """Initialize weights from description.

        Args:
            dct (dict): Dictionary description.

        Returns:
            CNBiasNbSetWeight.
        """
        if dct['type'] == 'linearly_equidistant':
            return cls.linearly_equidistant(weight_cn1=dct['weight_cn1'], weight_cn13=dct['weight_cn13'])
        if dct['type'] == 'geometrically_equidistant':
            return cls.geometrically_equidistant(weight_cn1=dct['weight_cn1'], weight_cn13=dct['weight_cn13'])
        if dct['type'] == 'explicit':
            return cls.explicit(cn_weights=dct['cn_weights'])
        raise RuntimeError('Cannot initialize Weights.')