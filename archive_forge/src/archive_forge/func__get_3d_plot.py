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
def _get_3d_plot(self, elements: list[Element], label_stable: bool | None, formulas_to_draw: list[str] | None, draw_formula_meshes: bool | None, draw_formula_lines: bool | None, formula_colors: list[str] | None, element_padding: float | None) -> Figure:
    """Returns a Plotly figure for a 3-dimensional chemical potential diagram."""
    if not formulas_to_draw:
        formulas_to_draw = []
    elem_indices = [self.elements.index(e) for e in elements]
    domains = self.domains.copy()
    domain_simplexes: dict[str, list[Simplex] | None] = {}
    draw_domains: dict[str, np.ndarray] = {}
    draw_comps = [Composition(formula).reduced_composition for formula in formulas_to_draw]
    annotations = []
    if element_padding and element_padding > 0:
        new_lims = self._get_new_limits_from_padding(domains, elem_indices, element_padding, self.default_min_limit)
    for formula, pts in domains.items():
        entry = self.entry_dict[formula]
        pts_3d = np.array(pts[:, elem_indices])
        if element_padding and element_padding > 0:
            for idx, new_lim in enumerate(new_lims):
                col = pts_3d[:, idx]
                pts_3d[:, idx] = np.where(np.isclose(col, self.default_min_limit), new_lim, col)
        contains_target_elems = set(entry.elements).issubset(elements)
        if formulas_to_draw and entry.composition.reduced_composition in draw_comps:
            domain_simplexes[formula] = None
            draw_domains[formula] = pts_3d
            if contains_target_elems:
                domains[formula] = pts_3d
            else:
                continue
        if not contains_target_elems:
            domain_simplexes[formula] = None
            continue
        simplexes, ann_loc = self._get_3d_domain_simplexes_and_ann_loc(pts_3d)
        anno_formula = formula
        if hasattr(entry, 'original_entry'):
            anno_formula = entry.original_entry.reduced_formula
        annotation = self._get_annotation(ann_loc, anno_formula)
        annotations.append(annotation)
        domain_simplexes[formula] = simplexes
    layout = plotly_layouts['default_layout_3d'].copy()
    layout['scene'].update(self._get_axis_layout_dict(elements))
    layout['scene']['annotations'] = None
    if label_stable:
        layout['scene']['annotations'] = annotations
    layout['scene_camera'] = {'eye': {'x': 5, 'y': 5, 'z': 5}, 'projection': {'type': 'orthographic'}, 'center': {'x': 0, 'y': 0, 'z': 0}}
    data = self._get_3d_domain_lines(domain_simplexes)
    if formulas_to_draw:
        for formula in formulas_to_draw:
            if formula not in domain_simplexes:
                warnings.warn(f'Specified formula to draw, {formula}, not found!')
    if draw_formula_lines:
        data.extend(self._get_3d_formula_lines(draw_domains, formula_colors))
    if draw_formula_meshes:
        data.extend(self._get_3d_formula_meshes(draw_domains, formula_colors))
    return Figure(data, layout)