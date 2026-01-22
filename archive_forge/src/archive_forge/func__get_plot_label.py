from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
def _get_plot_label(self, atoms, per_bond):
    all_labels = []
    for atoms_names in atoms:
        new = [self._split_string(atoms_names[0])[0], self._split_string(atoms_names[1])[0]]
        new.sort()
        string_here = f'{new[0]}-{new[1]}'
        all_labels.append(string_here)
    count = collections.Counter(all_labels)
    plotlabels = []
    for key, item in count.items():
        plotlabels.append(f'{item} x {key}')
    plotlabel = ', '.join(plotlabels)
    if per_bond:
        plotlabel = plotlabel + ' (per bond)'
    return plotlabel