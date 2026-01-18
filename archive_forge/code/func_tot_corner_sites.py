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
def tot_corner_sites(self):
    """
        Returns the number of vertices in the convex hull.
            Useful for identifying catalytically active sites.
        """
    return len(self.wulff_convex.vertices)