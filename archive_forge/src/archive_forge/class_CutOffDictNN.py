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
class CutOffDictNN(NearNeighbors):
    """
    A basic NN class using a dictionary of fixed cut-off distances.
    Only pairs of elements listed in the cut-off dictionary are considered
    during construction of the neighbor lists.

    Omit passing a dictionary for a Null/Empty NN class.
    """

    def __init__(self, cut_off_dict: dict | None=None) -> None:
        """
        Args:
            cut_off_dict (dict[str, float]): a dictionary
            of cut-off distances, e.g. {('Fe','O'): 2.0} for
            a maximum Fe-O bond length of 2 Angstroms.
            Bonds will only be created between pairs listed
            in the cut-off dictionary.
            If your structure is oxidation state decorated,
            the cut-off distances will have to explicitly include
            the oxidation state, e.g. {('Fe2+', 'O2-'): 2.0}.
        """
        self.cut_off_dict = cut_off_dict or {}
        self._max_dist = 0.0
        lookup_dict: dict[str, dict[str, float]] = defaultdict(dict)
        for (sp1, sp2), dist in self.cut_off_dict.items():
            lookup_dict[sp1][sp2] = dist
            lookup_dict[sp2][sp1] = dist
            if dist > self._max_dist:
                self._max_dist = dist
        self._lookup_dict = lookup_dict

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

    @classmethod
    def from_preset(cls, preset) -> Self:
        """
        Initialize a CutOffDictNN according to a preset set of cutoffs.

        Args:
            preset (str): A preset name. The list of supported presets are:
                - "vesta_2019": The distance cutoffs used by the VESTA visualisation program.

        Returns:
            A CutOffDictNN using the preset cut-off dictionary.
        """
        if preset == 'vesta_2019':
            cut_offs = loadfn(f'{module_dir}/vesta_cutoffs.yaml')
            return cls(cut_off_dict=cut_offs)
        raise ValueError(f'Unknown preset={preset!r}')

    def get_nn_info(self, structure: Structure, n: int) -> list[dict]:
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine near-neighbor sites.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one of which
                represents a coordinated site, its image location, and its weight.
        """
        site = structure[n]
        neighs_dists = structure.get_neighbors(site, self._max_dist)
        nn_info = []
        for nn in neighs_dists:
            n_site = nn
            dist = nn.nn_distance
            neigh_cut_off_dist = self._lookup_dict.get(site.species_string, {}).get(n_site.species_string, 0.0)
            if dist < neigh_cut_off_dist:
                nn_info.append({'site': n_site, 'image': self._get_image(structure, n_site), 'weight': dist, 'site_index': self._get_original_site(structure, n_site)})
        return nn_info