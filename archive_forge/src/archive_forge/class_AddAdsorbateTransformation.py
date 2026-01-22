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
class AddAdsorbateTransformation(AbstractTransformation):
    """Create adsorbate structures."""

    def __init__(self, adsorbate, selective_dynamics=False, height=0.9, mi_vec=None, repeat=None, min_lw=5.0, translate=True, reorient=True, find_args=None):
        """Use AdsorbateSiteFinder to add an adsorbate to a slab.

        Args:
            adsorbate (Molecule): molecule to add as adsorbate
            selective_dynamics (bool): flag for whether to assign
                non-surface sites as fixed for selective dynamics
            height (float): height criteria for selection of surface sites
            mi_vec : vector corresponding to the vector
                concurrent with the miller index, this enables use with
                slabs that have been reoriented, but the miller vector
                must be supplied manually
            repeat (3-tuple or list): repeat argument for supercell generation
            min_lw (float): minimum length and width of the slab, only used
                if repeat is None
            translate (bool): flag on whether to translate the molecule so
                that its CoM is at the origin prior to adding it to the surface
            reorient (bool): flag on whether or not to reorient adsorbate
                along the miller index
            find_args (dict): dictionary of arguments to be passed to the
                call to self.find_adsorption_sites, e.g. {"distance":2.0}
        """
        self.adsorbate = adsorbate
        self.selective_dynamics = selective_dynamics
        self.height = height
        self.mi_vec = mi_vec
        self.repeat = repeat
        self.min_lw = min_lw
        self.translate = translate
        self.reorient = reorient
        self.find_args = find_args

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False):
        """
        Args:
            structure: Must be a Slab structure
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures.

                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            Slab: with adsorbate
        """
        site_finder = AdsorbateSiteFinder(structure, selective_dynamics=self.selective_dynamics, height=self.height, mi_vec=self.mi_vec)
        structures = site_finder.generate_adsorption_structures(self.adsorbate, repeat=self.repeat, min_lw=self.min_lw, translate=self.translate, reorient=self.reorient, find_args=self.find_args)
        if not return_ranked_list:
            return structures[0]
        return [{'structure': structure} for structure in structures[:return_ranked_list]]

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True