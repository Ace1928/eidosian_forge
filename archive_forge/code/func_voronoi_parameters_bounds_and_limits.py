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
def voronoi_parameters_bounds_and_limits(self, isite, plot_type, max_dist):
    """
        Get the different boundaries and limits of the distance and angle factors for the given site.

        Args:
            isite: Index of the site.
            plot_type: Types of distance/angle parameters to get.
            max_dist: Maximum distance factor.

        Returns:
            Distance and angle bounds and limits.
        """
    if self.voronoi_list2[isite] is None:
        return None
    if plot_type is None:
        plot_type = {'distance_parameter': ('initial_inverse_opposite', None), 'angle_parameter': ('initial_opposite', None)}
    dd = [dist['min'] for dist in self.neighbors_normalized_distances[isite]]
    dd[0] = 1.0
    if plot_type['distance_parameter'][0] == 'initial_normalized':
        dd.append(max_dist)
        distance_bounds = np.array(dd)
        dist_limits = [1.0, max_dist]
    elif plot_type['distance_parameter'][0] == 'initial_inverse_opposite':
        ddinv = [1.0 / dist for dist in dd]
        ddinv.append(0.0)
        distance_bounds = np.array([1.0 - invdist for invdist in ddinv])
        dist_limits = [0.0, 1.0]
    elif plot_type['distance_parameter'][0] == 'initial_inverse3_opposite':
        ddinv = [1.0 / dist ** 3.0 for dist in dd]
        ddinv.append(0.0)
        distance_bounds = np.array([1.0 - invdist for invdist in ddinv])
        dist_limits = [0.0, 1.0]
    else:
        raise NotImplementedError(f'Plotting type {plot_type['distance_parameter']!r} for the distance is not implemented')
    if plot_type['angle_parameter'][0] == 'initial_normalized':
        aa = [0.0]
        aa.extend([ang['max'] for ang in self.neighbors_normalized_angles[isite]])
        angle_bounds = np.array(aa)
    elif plot_type['angle_parameter'][0] == 'initial_opposite':
        aa = [0.0]
        aa.extend([ang['max'] for ang in self.neighbors_normalized_angles[isite]])
        aa = [1.0 - ang for ang in aa]
        angle_bounds = np.array(aa)
    else:
        raise NotImplementedError(f'Plotting type {plot_type['angle_parameter']!r} for the angle is not implemented')
    ang_limits = [0.0, 1.0]
    return {'distance_bounds': distance_bounds, 'distance_limits': dist_limits, 'angle_bounds': angle_bounds, 'angle_limits': ang_limits}