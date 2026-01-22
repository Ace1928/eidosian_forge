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
def _get_3d_formula_meshes(draw_domains: dict[str, np.ndarray], formula_colors: list[str] | None) -> list[Mesh3d]:
    """
        Returns a list of Mesh3d objects for the domains specified by the
        user (i.e., draw_domains).
        """
    meshes = []
    if formula_colors is None:
        formula_colors = px.colors.qualitative.Dark2
    for idx, (formula, coords) in enumerate(draw_domains.items()):
        points_3d = coords[:, :3]
        mesh = Mesh3d(x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2], alphahull=0, showlegend=True, lighting={'fresnel': 1.0}, color=formula_colors[idx], name=f'{formula} (mesh)', opacity=0.13)
        meshes.append(mesh)
    return meshes