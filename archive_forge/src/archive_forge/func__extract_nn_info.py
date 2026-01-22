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
def _extract_nn_info(self, structure: Structure, nns):
    """Given Voronoi NNs, extract the NN info in the form needed by NearestNeighbors.

        Args:
            structure (Structure): Structure being evaluated
            nns ([dicts]): Nearest neighbor information for a structure

        Returns:
            list[tuple[PeriodicSite, np.ndarray, float]]: tuples of the form
                (site, image, weight). See nn_info.
        """
    targets = structure.elements if self.targets is None else self.targets
    siw = []
    max_weight = max((nn[self.weight] for nn in nns.values()))
    for nstats in nns.values():
        site = nstats['site']
        if nstats[self.weight] > self.tol * max_weight and _is_in_targets(site, targets):
            nn_info = {'site': site, 'image': self._get_image(structure, site), 'weight': nstats[self.weight] / max_weight, 'site_index': self._get_original_site(structure, site)}
            if self.extra_nn_info:
                poly_info = nstats
                del poly_info['site']
                nn_info['poly_info'] = poly_info
            siw.append(nn_info)
    return siw