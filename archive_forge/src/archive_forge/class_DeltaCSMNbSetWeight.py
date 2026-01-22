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
class DeltaCSMNbSetWeight(NbSetWeight):
    """Weight of neighbors set based on the differences of CSM."""
    SHORT_NAME = 'DeltaCSMWeight'
    DEFAULT_EFFECTIVE_CSM_ESTIMATOR = dict(function='power2_inverse_decreasing', options={'max_csm': 8.0})
    DEFAULT_SYMMETRY_MEASURE_TYPE = 'csm_wcs_ctwcc'
    DEFAULT_WEIGHT_ESTIMATOR = dict(function='smootherstep', options={'delta_csm_min': 0.5, 'delta_csm_max': 3.0})

    def __init__(self, effective_csm_estimator=DEFAULT_EFFECTIVE_CSM_ESTIMATOR, weight_estimator=DEFAULT_WEIGHT_ESTIMATOR, delta_cn_weight_estimators=None, symmetry_measure_type=DEFAULT_SYMMETRY_MEASURE_TYPE):
        """Initialize DeltaCSMNbSetWeight.

        Args:
            effective_csm_estimator: Ratio function used for the effective CSM (comparison between neighbors sets).
            weight_estimator: Weight estimator within a given neighbors set.
            delta_cn_weight_estimators: Specific weight estimators for specific cn
            symmetry_measure_type: Type of symmetry measure to be used.
        """
        self.effective_csm_estimator = effective_csm_estimator
        self.effective_csm_estimator_rf = CSMInfiniteRatioFunction.from_dict(effective_csm_estimator)
        self.weight_estimator = weight_estimator
        if self.weight_estimator is not None:
            self.weight_estimator_rf = DeltaCSMRatioFunction.from_dict(weight_estimator)
        self.delta_cn_weight_estimators = delta_cn_weight_estimators
        self.delta_cn_weight_estimators_rfs = {}
        if delta_cn_weight_estimators is not None:
            for delta_cn, dcn_w_estimator in delta_cn_weight_estimators.items():
                self.delta_cn_weight_estimators_rfs[delta_cn] = DeltaCSMRatioFunction.from_dict(dcn_w_estimator)
        self.symmetry_measure_type = symmetry_measure_type
        self.max_effective_csm = self.effective_csm_estimator['options']['max_csm']

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
        effcsm = get_effective_csm(nb_set=nb_set, cn_map=cn_map, structure_environments=structure_environments, additional_info=additional_info, symmetry_measure_type=self.symmetry_measure_type, max_effective_csm=self.max_effective_csm, effective_csm_estimator_ratio_function=self.effective_csm_estimator_rf)
        cn = cn_map[0]
        isite = nb_set.isite
        delta_csm = delta_csm_cn_map2 = None
        nb_set_weight = 1
        for cn2, nb_sets in structure_environments.neighbors_sets[isite].items():
            if cn2 < cn:
                continue
            for inb_set2, nb_set2 in enumerate(nb_sets):
                if cn == cn2:
                    continue
                effcsm2 = get_effective_csm(nb_set=nb_set2, cn_map=(cn2, inb_set2), structure_environments=structure_environments, additional_info=additional_info, symmetry_measure_type=self.symmetry_measure_type, max_effective_csm=self.max_effective_csm, effective_csm_estimator_ratio_function=self.effective_csm_estimator_rf)
                this_delta_csm = effcsm2 - effcsm
                if cn2 == cn:
                    if this_delta_csm < 0:
                        set_info(additional_info=additional_info, field='delta_csms', isite=isite, cn_map=cn_map, value=this_delta_csm)
                        set_info(additional_info=additional_info, field='delta_csms_weights', isite=isite, cn_map=cn_map, value=0)
                        set_info(additional_info=additional_info, field='delta_csms_cn_map2', isite=isite, cn_map=cn_map, value=(cn2, inb_set2))
                        return 0
                else:
                    dcn = cn2 - cn
                    if dcn in self.delta_cn_weight_estimators_rfs:
                        this_delta_csm_weight = self.delta_cn_weight_estimators_rfs[dcn].evaluate(this_delta_csm)
                    else:
                        this_delta_csm_weight = self.weight_estimator_rf.evaluate(this_delta_csm)
                    if this_delta_csm_weight < nb_set_weight:
                        delta_csm = this_delta_csm
                        delta_csm_cn_map2 = (cn2, inb_set2)
                        nb_set_weight = this_delta_csm_weight
        set_info(additional_info=additional_info, field='delta_csms', isite=isite, cn_map=cn_map, value=delta_csm)
        set_info(additional_info=additional_info, field='delta_csms_weights', isite=isite, cn_map=cn_map, value=nb_set_weight)
        set_info(additional_info=additional_info, field='delta_csms_cn_map2', isite=isite, cn_map=cn_map, value=delta_csm_cn_map2)
        return nb_set_weight

    def __eq__(self, other: object) -> bool:
        needed_attrs = ['effective_csm_estimator', 'weight_estimator', 'delta_cn_weight_estimators', 'symmetry_measure_type']
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        return all((getattr(self, attr) == getattr(other, attr) for attr in needed_attrs))

    @classmethod
    def delta_cn_specifics(cls, delta_csm_mins=None, delta_csm_maxs=None, function='smootherstep', symmetry_measure_type='csm_wcs_ctwcc', effective_csm_estimator=DEFAULT_EFFECTIVE_CSM_ESTIMATOR):
        """Initialize DeltaCSMNbSetWeight from specific coordination number differences.

        Args:
            delta_csm_mins: Minimums for each coordination number.
            delta_csm_maxs: Maximums for each coordination number.
            function: Ratio function used.
            symmetry_measure_type: Type of symmetry measure to be used.
            effective_csm_estimator: Ratio function used for the effective CSM (comparison between neighbors sets).

        Returns:
            DeltaCSMNbSetWeight.
        """
        if delta_csm_mins is None or delta_csm_maxs is None:
            delta_cn_weight_estimators = {dcn: {'function': function, 'options': {'delta_csm_min': 0.25 + dcn * 0.25, 'delta_csm_max': 5.0 + dcn * 0.25}} for dcn in range(1, 13)}
        else:
            delta_cn_weight_estimators = {dcn: {'function': function, 'options': {'delta_csm_min': delta_csm_mins[dcn - 1], 'delta_csm_max': delta_csm_maxs[dcn - 1]}} for dcn in range(1, 13)}
        return cls(effective_csm_estimator=effective_csm_estimator, weight_estimator={'function': function, 'options': {'delta_csm_min': delta_cn_weight_estimators[12]['options']['delta_csm_min'], 'delta_csm_max': delta_cn_weight_estimators[12]['options']['delta_csm_max']}}, delta_cn_weight_estimators=delta_cn_weight_estimators, symmetry_measure_type=symmetry_measure_type)

    def as_dict(self):
        """
        MSONable dict.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'effective_csm_estimator': self.effective_csm_estimator, 'weight_estimator': self.weight_estimator, 'delta_cn_weight_estimators': self.delta_cn_weight_estimators, 'symmetry_measure_type': self.symmetry_measure_type}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize from dict.

        Args:
            dct (dict): Dict representation of DeltaCSMNbSetWeight.

        Returns:
            DeltaCSMNbSetWeight.
        """
        return cls(effective_csm_estimator=dct['effective_csm_estimator'], weight_estimator=dct['weight_estimator'], delta_cn_weight_estimators={int(dcn): dcn_estimator for dcn, dcn_estimator in dct['delta_cn_weight_estimators'].items()} if dct.get('delta_cn_weight_estimators') is not None else None, symmetry_measure_type=dct['symmetry_measure_type'])