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
def add_strategy_visualization_to_subplot(self, subplot, visualization_options=None, plot_type=None):
    """Add a visual of the strategy on a distance-angle plot.

        Args:
            subplot: Axes object onto the visual should be added.
            visualization_options: Options for the visual.
            plot_type: Type of distance-angle plot.
        """
    subplot.plot(self._distance_cutoff, self._angle_cutoff, 'o', markeredgecolor=None, markerfacecolor='w', markersize=12)
    subplot.plot(self._distance_cutoff, self._angle_cutoff, 'x', linewidth=2, markersize=12)