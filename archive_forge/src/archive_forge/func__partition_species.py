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