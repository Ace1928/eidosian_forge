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
def tot_edges(self):
    """
        Returns the number of edges in the convex hull.
            Useful for identifying catalytically active sites.
        """
    all_edges = []
    for facet in self.facets:
        edges = []
        pt = self.get_line_in_facet(facet)
        lines = []
        for idx in range(len(pt)):
            if idx == len(pt) / 2:
                break
            lines.append(tuple(sorted((tuple(pt[idx * 2]), tuple(pt[idx * 2 + 1])))))
        for p in lines:
            if p not in all_edges:
                edges.append(p)
        all_edges.extend(edges)
    return len(all_edges)