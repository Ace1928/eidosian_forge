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
def _get_original_site(structure: Structure, site: Site) -> int:
    """Private convenience method for get_nn_info,
        gives original site index from ProvidedPeriodicSite.
        """
    if isinstance(site, PeriodicNeighbor):
        return site.index
    if isinstance(structure, (IStructure, Structure)):
        for idx, struc_site in enumerate(structure):
            if site.is_periodic_image(struc_site):
                return idx
    else:
        for idx, struc_site in enumerate(structure):
            if site == struc_site:
                return idx
    raise ValueError('Site not found in structure')