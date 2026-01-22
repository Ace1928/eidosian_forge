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
class BrunnerNNRelative(NearNeighbors):
    """
    Determine coordination number using Brunner's algorithm which counts the
    atoms that are within the largest gap in differences in real space
    interatomic distances. This algorithm uses Brunner's method of
    of largest relative gap in interatomic distances.
    """

    def __init__(self, tol: float=0.0001, cutoff=8.0) -> None:
        """
        Args:
            tol (float): tolerance parameter for bond determination
                (default: 1E-4).
            cutoff (float): cutoff radius in Angstrom to look for near-neighbor
                atoms. Defaults to 8.0.
        """
        self.tol = tol
        self.cutoff = cutoff

    @property
    def structures_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return True

    @property
    def molecules_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        return False

    def get_nn_info(self, structure: Structure, n: int):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near-neighbor sites.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location,
                and its weight.
        """
        site = structure[n]
        neighs_dists = structure.get_neighbors(site, self.cutoff)
        ds = sorted((idx.nn_distance for idx in neighs_dists))
        ns = [ds[idx + 1] / ds[idx] for idx in range(len(ds) - 1)]
        d_max = ds[ns.index(max(ns))]
        siw = []
        for nn in neighs_dists:
            s, dist = (nn, nn.nn_distance)
            if dist < d_max + self.tol:
                w = ds[0] / dist
                siw.append({'site': s, 'image': self._get_image(structure, s), 'weight': w, 'site_index': self._get_original_site(structure, s)})
        return siw