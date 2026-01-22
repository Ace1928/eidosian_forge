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
@staticmethod
def _semicircle_integral(dist_bins, idx):
    """
        An internal method to get an integral between two bounds of a unit
        semicircle. Used in algorithm to determine bond probabilities.

        Args:
            dist_bins: (float) list of all possible bond weights
            idx: (float) index of starting bond weight

        Returns:
            float: integral of portion of unit semicircle
        """
    radius = 1
    x1 = dist_bins[idx]
    x2 = dist_bins[idx + 1]
    if dist_bins[idx] == 1:
        area1 = 0.25 * math.pi * radius ** 2
    else:
        area1 = 0.5 * (x1 * math.sqrt(radius ** 2 - x1 ** 2) + radius ** 2 * math.atan(x1 / math.sqrt(radius ** 2 - x1 ** 2)))
    area2 = 0.5 * (x2 * math.sqrt(radius ** 2 - x2 ** 2) + radius ** 2 * math.atan(x2 / math.sqrt(radius ** 2 - x2 ** 2)))
    return (area1 - area2) / (0.25 * math.pi * radius ** 2)