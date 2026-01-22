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
def _get_min_entries_and_el_refs(entries: list[PDEntry]) -> tuple[list[PDEntry], dict[Element, PDEntry]]:
    """
        Returns a list of the minimum-energy entries at each composition and the
        entries corresponding to the elemental references.
        """
    el_refs = {}
    min_entries = []
    for formula, group in groupby(entries, key=lambda e: e.reduced_formula):
        comp = Composition(formula)
        min_entry = min(group, key=lambda e: e.energy_per_atom)
        if comp.is_element:
            el_refs[comp.elements[0]] = min_entry
        min_entries.append(min_entry)
    return (min_entries, el_refs)