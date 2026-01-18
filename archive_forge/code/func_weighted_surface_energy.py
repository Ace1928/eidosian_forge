from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
@property
def weighted_surface_energy(self) -> float:
    """
        Returns:
            sum(surface_energy_hkl * area_hkl)/ sum(area_hkl).
        """
    return self.total_surface_energy / self.surface_area