from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def add_site_types() -> None:
    """Add Wyckoff symbols and equivalent sites to the initial structure."""
    if 'bulk_wyckoff' not in initial_structure.site_properties or 'bulk_equivalent' not in initial_structure.site_properties:
        spg_analyzer = SpacegroupAnalyzer(initial_structure)
        initial_structure.add_site_property('bulk_wyckoff', spg_analyzer.get_symmetry_dataset()['wyckoffs'])
        initial_structure.add_site_property('bulk_equivalent', spg_analyzer.get_symmetry_dataset()['equivalent_atoms'].tolist())