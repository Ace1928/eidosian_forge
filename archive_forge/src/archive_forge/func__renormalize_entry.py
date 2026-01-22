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
def _renormalize_entry(entry: PDEntry, renormalization_energy_per_atom: float) -> PDEntry:
    """Regenerate the input entry with an energy per atom decreased by renormalization_energy_per_atom."""
    renormalized_entry_dict = entry.as_dict()
    renormalized_entry_dict['energy'] = entry.energy - renormalization_energy_per_atom * sum(entry.composition.values())
    return PDEntry.from_dict(renormalized_entry_dict)