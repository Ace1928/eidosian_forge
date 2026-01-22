from __future__ import annotations
import logging
import math
import warnings
from fractions import Fraction
from itertools import groupby, product
from math import gcd
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from monty.dev import requires
from monty.fractions import lcm
from monty.json import MSONable
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.energy_models import SymmetryModel
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.structure_matcher import SpinComparator, StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.command_line.enumlib_caller import EnumError, EnumlibAdaptor
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core import DummySpecies, Element, Species, Structure, get_el_sp
from pymatgen.core.surface import SlabGenerator
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.icet import IcetSQS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
from pymatgen.transformations.transformation_abc import AbstractTransformation
class DisorderOrderedTransformation(AbstractTransformation):
    """Not to be confused with OrderDisorderedTransformation,
    this transformation attempts to obtain a
    *disordered* structure from an input ordered structure.
    This may or may not be physically plausible, further
    inspection of the returned structures is advised.
    The main purpose for this transformation is for structure
    matching to crystal prototypes for structures that have
    been derived from a parent prototype structure by
    substitutions or alloying additions.
    """

    def __init__(self, max_sites_to_merge=2):
        """
        Args:
            max_sites_to_merge: only merge this number of sites together.
        """
        self.max_sites_to_merge = max_sites_to_merge

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False):
        """
        Args:
            structure: ordered structure
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures.

                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            Transformed disordered structure(s)
        """
        if not structure.is_ordered:
            raise ValueError('This transformation is for disordered structures only.')
        partitions = self._partition_species(structure.composition, max_components=self.max_sites_to_merge)
        disorder_mappings = self._get_disorder_mappings(structure.composition, partitions)
        disordered_structures = []
        for mapping in disorder_mappings:
            disordered_structure = structure.copy()
            disordered_structure.replace_species(mapping)
            disordered_structures.append({'structure': disordered_structure, 'mapping': mapping})
        if len(disordered_structures) == 0:
            return None
        if not return_ranked_list:
            return disordered_structures[0]['structure']
        if len(disordered_structures) > return_ranked_list:
            disordered_structures = disordered_structures[0:return_ranked_list]
        return disordered_structures

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True

    @staticmethod
    def _partition_species(composition, max_components=2):
        """Private method to split a list of species into
        various partitions.
        """

        def _partition(collection):
            if len(collection) == 1:
                yield [collection]
                return
            first = collection[0]
            for smaller in _partition(collection[1:]):
                for n, subset in enumerate(smaller):
                    yield (smaller[:n] + [[first, *subset]] + smaller[n + 1:])
                yield [[first], *smaller]

        def _sort_partitions(partitions_to_sort):
            """Sort partitions by those we want to check first
            (typically, merging two sites into one is the one to try first).
            """
            partition_indices = [(idx, [len(p) for p in partition]) for idx, partition in enumerate(partitions_to_sort)]
            partition_indices = sorted(partition_indices, key=lambda x: (max(x[1]), -len(x[1])))
            partition_indices = [x for x in partition_indices if max(x[1]) <= max_components]
            partition_indices.pop(0)
            return [partitions_to_sort[x[0]] for x in partition_indices]
        collection = list(composition)
        partitions = list(_partition(collection))
        return _sort_partitions(partitions)

    @staticmethod
    def _get_disorder_mappings(composition, partitions):
        """Private method to obtain the mapping to create
        a disordered structure from a given partition.
        """

        def _get_replacement_dict_from_partition(partition):
            dct = {}
            for sp_list in partition:
                if len(sp_list) > 1:
                    total_occ = sum((composition[sp] for sp in sp_list))
                    merged_comp = {sp: composition[sp] / total_occ for sp in sp_list}
                    for sp in sp_list:
                        dct[sp] = merged_comp
            return dct
        return [_get_replacement_dict_from_partition(p) for p in partitions]