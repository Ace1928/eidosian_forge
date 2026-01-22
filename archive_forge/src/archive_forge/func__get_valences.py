from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
def _get_valences(self):
    """Computes ionic valences of elements for all sites in the structure."""
    try:
        bv = BVAnalyzer()
        self._structure = bv.get_oxi_state_decorated_structure(self._structure)
        valences = bv.get_valences(self._structure)
    except Exception:
        try:
            bv = BVAnalyzer(symm_tol=0)
            self._structure = bv.get_oxi_state_decorated_structure(self._structure)
            valences = bv.get_valences(self._structure)
        except Exception:
            valences = []
            for site in self._structure:
                if len(site.specie.common_oxidation_states) > 0:
                    valences.append(site.specie.common_oxidation_states[0])
                else:
                    valences.append(0)
            if sum(valences):
                valences = [0] * len(self._structure)
            else:
                self._structure.add_oxidation_state_by_site(valences)
    return valences