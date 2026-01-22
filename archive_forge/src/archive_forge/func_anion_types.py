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
@property
def anion_types(self):
    """
        Return the types of anions present in crystal structure as a set

        Returns:
            set[Element]: describing anions in the crystal structure.
        """
    if self.valences is None:
        raise ValueError('No cations and anions defined')
    anion_species = []
    for site, val in zip(self.structure, self.valences):
        if val < 0.0:
            anion_species.append(site.specie)
    return set(anion_species)