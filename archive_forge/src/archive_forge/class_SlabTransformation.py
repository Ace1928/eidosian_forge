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
class SlabTransformation(AbstractTransformation):
    """A transformation that creates a slab from a structure."""

    def __init__(self, miller_index, min_slab_size, min_vacuum_size, lll_reduce=False, center_slab=False, in_unit_planes=False, primitive=True, max_normal_search=None, shift=0, tol=0.1):
        """
        Args:
            miller_index (3-tuple or list): miller index of slab
            min_slab_size (float): minimum slab size in angstroms
            min_vacuum_size (float): minimum size of vacuum
            lll_reduce (bool): whether to apply LLL reduction
            center_slab (bool): whether to center the slab
            primitive (bool): whether to reduce slabs to most primitive cell
            in_unit_planes (bool): Whether to set min_slab_size and min_vac_size
                in units of hkl planes (True) or Angstrom (False, the default). Setting in
                units of planes is useful for ensuring some slabs have a certain n_layer of
                atoms. e.g. for Cs (100), a 10 Ang slab will result in a slab with only 2
                layer of atoms, whereas Fe (100) will have more layer of atoms. By using units
                of hkl planes instead, we ensure both slabs have the same number of atoms. The
                slab thickness will be in min_slab_size/math.ceil(self._proj_height/dhkl)
                multiples of oriented unit cells.
            max_normal_search (int): maximum index to include in linear
                combinations of indices to find c lattice vector orthogonal
                to slab surface
            shift (float): shift to get termination
            tol (float): tolerance for primitive cell finding.
        """
        self.miller_index = miller_index
        self.min_slab_size = min_slab_size
        self.min_vacuum_size = min_vacuum_size
        self.lll_reduce = lll_reduce
        self.center_slab = center_slab
        self.in_unit_planes = in_unit_planes
        self.primitive = primitive
        self.max_normal_search = max_normal_search
        self.shift = shift
        self.tol = tol

    def apply_transformation(self, structure: Structure):
        """Applies the transformation.

        Args:
            structure: Input Structure

        Returns:
            Slab Structures.
        """
        sg = SlabGenerator(structure, self.miller_index, self.min_slab_size, self.min_vacuum_size, self.lll_reduce, self.center_slab, self.in_unit_planes, self.primitive, self.max_normal_search)
        return sg.get_slab(self.shift, self.tol)

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False