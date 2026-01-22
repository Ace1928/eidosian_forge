from a list of entries within a chemical system containing 2 or more elements. The
from __future__ import annotations
import json
import os
import warnings
from functools import lru_cache
from itertools import groupby
from typing import TYPE_CHECKING
import numpy as np
import plotly.express as px
from monty.json import MSONable
from plotly.graph_objects import Figure, Mesh3d, Scatter, Scatter3d
from scipy.spatial import ConvexHull, HalfspaceIntersection
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition, Element
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.string import htmlify
@staticmethod
def _get_new_limits_from_padding(domains: dict[str, np.ndarray], elem_indices: list[int], element_padding: float, default_min_limit: float):
    """
        Gets new minimum limits for each element by subtracting specified padding
        from the minimum for each axis found in any of the domains.
        """
    all_pts = np.vstack(list(domains.values()))
    new_lims = []
    for el in elem_indices:
        pts = all_pts[:, el]
        new_lim = pts[~np.isclose(pts, default_min_limit)].min() - element_padding
        new_lims.append(new_lim)
    return new_lims