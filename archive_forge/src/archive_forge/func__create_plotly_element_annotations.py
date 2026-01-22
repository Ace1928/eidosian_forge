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
def _create_plotly_element_annotations(self):
    """
        Creates terminal element annotations for Plotly phase diagrams. This method does
        not apply to ternary_2d plots.

        Functionality is included for phase diagrams with non-elemental endmembers
        (as is true for grand potential phase diagrams).

        Returns:
            List of annotation dicts.
        """
    annotations_list = []
    x, y, z = (None, None, None)
    if self._dim == 3 and self.ternary_style == '2d':
        return None
    for coords, entry in self.pd_plot_data[1].items():
        if not entry.composition.is_element:
            continue
        x, y = (coords[0], coords[1])
        if self._dim == 3:
            z = self._pd.get_form_energy_per_atom(entry)
        elif self._dim == 4:
            z = coords[2]
        if entry.composition.is_element:
            clean_formula = str(entry.elements[0])
            if hasattr(entry, 'original_entry'):
                orig_comp = entry.original_entry.composition
                clean_formula = htmlify(orig_comp.reduced_formula)
            font_dict = {'color': '#000000', 'size': 24.0}
            opacity = 1.0
        offset = 0.03 if self._dim == 2 else 0.06
        if x < 0.4:
            x -= offset
        elif x > 0.6:
            x += offset
        if y < 0.1:
            y -= offset
        elif y > 0.8:
            y += offset
        if self._dim == 4 and z > 0.8:
            z += offset
        annotation = plotly_layouts['default_annotation_layout'].copy()
        annotation.update(x=x, y=y, font=font_dict, text=clean_formula, opacity=opacity)
        if self._dim in (3, 4):
            for d in ['xref', 'yref']:
                annotation.pop(d)
                if self._dim == 3:
                    annotation.update({'x': y, 'y': x})
                    if entry.composition.is_element:
                        z = 0.9 * self._min_energy
            annotation['z'] = z
        annotations_list.append(annotation)
    if self._dim == 3:
        annotations_list.append({'x': 1, 'y': 1, 'z': 0, 'opacity': 0, 'text': ''})
    return annotations_list