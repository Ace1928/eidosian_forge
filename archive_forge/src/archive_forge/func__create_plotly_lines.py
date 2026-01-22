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
def _create_plotly_lines(self):
    """
        Create Plotly scatter plots containing line traces of phase diagram facets.

        Returns:
            Either a go.Scatter (binary), go.Scatterternary (ternary_2d), or
            go.Scatter3d plot (ternary_3d, quaternary)
        """
    line_plot = None
    x, y, z, energies = ([], [], [], [])
    pd = self._pd
    plot_args = {'mode': 'lines', 'hoverinfo': 'none', 'line': {'color': 'black', 'width': 4.0}, 'showlegend': False}
    if self._dim == 3 and self.ternary_style == '2d':
        plot_args['line']['width'] = 1.5
        el_a, el_b, el_c = pd.elements
        for line in uniquelines(pd.facets):
            e0 = pd.qhull_entries[line[0]]
            e1 = pd.qhull_entries[line[1]]
            x += [e0.composition[el_a], e1.composition[el_a], None]
            y += [e0.composition[el_b], e1.composition[el_b], None]
            z += [e0.composition[el_c], e1.composition[el_c], None]
    else:
        for line in self.pd_plot_data[0]:
            x += [*line[0], None]
            y += [*line[1], None]
            if self._dim == 3:
                form_enes = [self._pd.get_form_energy_per_atom(self.pd_plot_data[1][coord]) for coord in zip(line[0], line[1])]
                z += [*form_enes, None]
            elif self._dim == 4:
                form_enes = [self._pd.get_form_energy_per_atom(self.pd_plot_data[1][coord]) for coord in zip(line[0], line[1], line[2])]
                energies += [*form_enes, None]
                z += [*line[2], None]
    if self._dim == 2:
        line_plot = go.Scatter(x=x, y=y, **plot_args)
    elif self._dim == 3 and self.ternary_style == '2d':
        line_plot = go.Scatterternary(a=x, b=y, c=z, **plot_args)
    elif self._dim == 3 and self.ternary_style == '3d':
        line_plot = go.Scatter3d(x=y, y=x, z=z, **plot_args)
    elif self._dim == 4:
        plot_args['line']['width'] = 1.5
        line_plot = go.Scatter3d(x=x, y=y, z=z, **plot_args)
    return line_plot