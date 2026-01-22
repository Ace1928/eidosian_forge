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
def _get_2d_plot(self, elements: list[Element], label_stable: bool | None, element_padding: float | None) -> Figure:
    """Returns a Plotly figure for a 2-dimensional chemical potential diagram."""
    domains = self.domains.copy()
    elem_indices = [self.elements.index(e) for e in elements]
    annotations = []
    draw_domains = {}
    if element_padding is not None and element_padding > 0:
        new_lims = self._get_new_limits_from_padding(domains, elem_indices, element_padding, self.default_min_limit)
    for formula, pts in domains.items():
        formula_elems = set(Composition(formula).elements)
        if not formula_elems.issubset(elements):
            continue
        pts_2d = np.array(pts[:, elem_indices])
        if element_padding is not None and element_padding > 0:
            for idx, new_lim in enumerate(new_lims):
                col = pts_2d[:, idx]
                pts_2d[:, idx] = np.where(np.isclose(col, self.default_min_limit), new_lim, col)
        entry = self.entry_dict[formula]
        anno_formula = formula
        if hasattr(entry, 'original_entry'):
            anno_formula = entry.original_entry.reduced_formula
        center = pts_2d.mean(axis=0)
        normal = get_2d_orthonormal_vector(pts_2d)
        ann_loc = center + 0.25 * normal
        annotation = self._get_annotation(ann_loc, anno_formula)
        annotations.append(annotation)
        draw_domains[formula] = pts_2d
    layout = plotly_layouts['default_layout_2d'].copy()
    layout.update(self._get_axis_layout_dict(elements))
    if label_stable:
        layout['annotations'] = annotations
    data = self._get_2d_domain_lines(draw_domains)
    return Figure(data, layout)