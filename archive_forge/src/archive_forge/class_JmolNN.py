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
class JmolNN(NearNeighbors):
    """
    Determine near-neighbor sites and coordination number using an emulation
    of Jmol's default autoBond() algorithm. This version of the algorithm
    does not take into account any information regarding known charge
    states.
    """

    def __init__(self, tol: float=0.45, min_bond_distance: float=0.4, el_radius_updates: dict[SpeciesLike, float] | None=None):
        """
        Args:
            tol (float): tolerance parameter for bond determination
                Defaults to 0.56.
            min_bond_distance (float): minimum bond distance for consideration
                Defaults to 0.4.
            el_radius_updates (dict): symbol->float to override default atomic
                radii table values.
        """
        self.tol = tol
        self.min_bond_distance = min_bond_distance
        bonds_file = f'{module_dir}/bonds_jmol_ob.yaml'
        with open(bonds_file) as file:
            yaml = YAML()
            self.el_radius = yaml.load(file)
        if el_radius_updates:
            self.el_radius.update(el_radius_updates)

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
        return True

    @property
    def extend_structure_molecules(self) -> bool:
        """
        Boolean property: Do Molecules need to be converted to Structures to use
        this NearNeighbors class? Note: this property is not defined for classes
        for which molecules_allowed is False.
        """
        return True

    def get_max_bond_distance(self, el1_sym, el2_sym):
        """
        Use Jmol algorithm to determine bond length from atomic parameters

        Args:
            el1_sym (str): symbol of atom 1
            el2_sym (str): symbol of atom 2.

        Returns:
            float: max bond length
        """
        return sqrt((self.el_radius[el1_sym] + self.el_radius[el2_sym] + self.tol) ** 2)

    def get_nn_info(self, structure: Structure, n: int):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n using the bond identification
        algorithm underlying Jmol.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near
                neighbors.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a neighbor site, its image location,
                and its weight.
        """
        site = structure[n]
        bonds = {}
        for el in structure.elements:
            bonds[site.specie, el] = self.get_max_bond_distance(site.specie.symbol, el.symbol)
        max_rad = max(bonds.values()) + self.tol
        min_rad = min(bonds.values())
        siw = []
        for nn in structure.get_neighbors(site, max_rad):
            dist = nn.nn_distance
            if dist <= bonds[site.specie, nn.specie] and nn.nn_distance > self.min_bond_distance:
                weight = min_rad / dist
                siw.append({'site': nn, 'image': self._get_image(structure, nn), 'weight': weight, 'site_index': self._get_original_site(structure, nn)})
        return siw