from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def _create_plotly_uncertainty_shading(self, stable_marker_plot):
    """
        Creates shaded uncertainty region for stable entries. Currently only works
        for binary (dim=2) phase diagrams.

        Args:
            stable_marker_plot: go.Scatter object with stable markers and their
            error bars.

        Returns:
            Plotly go.Scatter object with uncertainty window shading.
        """
    uncertainty_plot = None
    x = stable_marker_plot.x
    y = stable_marker_plot.y
    transformed = False
    if hasattr(self._pd, 'original_entries') or hasattr(self._pd, 'chempots'):
        transformed = True
    if self._dim == 2:
        error = stable_marker_plot.error_y['array']
        points = np.append(x, [y, error]).reshape(3, -1).T
        points = points[points[:, 0].argsort()]
        outline = points[:, :2].copy()
        outline[:, 1] = outline[:, 1] + points[:, 2]
        last = -1
        if transformed:
            last = None
        flipped_points = np.flip(points[:last, :].copy(), axis=0)
        flipped_points[:, 1] = flipped_points[:, 1] - flipped_points[:, 2]
        outline = np.vstack((outline, flipped_points[:, :2]))
        uncertainty_plot = go.Scatter(x=outline[:, 0], y=outline[:, 1], name='Uncertainty (window)', fill='toself', mode='lines', line={'width': 0}, fillcolor='lightblue', hoverinfo='skip', opacity=0.4)
    return uncertainty_plot