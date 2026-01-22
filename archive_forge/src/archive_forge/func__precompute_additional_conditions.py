from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.math_utils import normal_cdf_step
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
def _precompute_additional_conditions(self, ivoronoi, voronoi, valences):
    additional_conditions = {ac: [] for ac in self.additional_conditions}
    for _, vals in voronoi:
        for ac in self.additional_conditions:
            additional_conditions[ac].append(self.AC.check_condition(condition=ac, structure=self.structure, parameters={'valences': valences, 'neighbor_index': vals['index'], 'site_index': ivoronoi}))
    return additional_conditions