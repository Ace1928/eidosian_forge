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
class MagOrderParameterConstraint(MSONable):
    """This class can be used to supply MagOrderingTransformation
    to just a specific subset of species or sites that satisfy the
    provided constraints. This can be useful for setting an order
    parameters for, for example, ferrimagnetic structures which
    might order on certain motifs, with the global order parameter
    dependent on how many sites satisfy that motif.
    """

    def __init__(self, order_parameter, species_constraints=None, site_constraint_name=None, site_constraints=None):
        """
        Args:
            order_parameter (float): any number from 0.0 to 1.0,
                typically 0.5 (antiferromagnetic) or 1.0 (ferromagnetic)
            species_constraints (list): str or list of strings
                of Species symbols that the constraint should apply to
            site_constraint_name (str): name of the site property
                that the constraint should apply to, e.g. "coordination_no"
            site_constraints (list): list of values of the site
                property that the constraints should apply to.
        """
        if site_constraints and site_constraints != [None] and (not site_constraint_name):
            raise ValueError('Specify the name of the site constraint.')
        if not site_constraints and site_constraint_name:
            raise ValueError('Please specify some site constraints.')
        if not isinstance(species_constraints, list):
            species_constraints = [species_constraints]
        if not isinstance(site_constraints, list):
            site_constraints = [site_constraints]
        if order_parameter > 1 or order_parameter < 0:
            raise ValueError('Order parameter must lie between 0 and 1')
        if order_parameter != 0.5:
            warnings.warn('Use care when using a non-standard order parameter, though it can be useful in some cases it can also lead to unintended behavior. Consult documentation.')
        self.order_parameter = order_parameter
        self.species_constraints = species_constraints
        self.site_constraint_name = site_constraint_name
        self.site_constraints = site_constraints

    def satisfies_constraint(self, site):
        """Checks if a periodic site satisfies the constraint."""
        if not site.is_ordered:
            return False
        satisfies_constraints = self.species_constraints and str(site.specie) in self.species_constraints
        if self.site_constraint_name and self.site_constraint_name in site.properties:
            prop = site.properties[self.site_constraint_name]
            satisfies_constraints = prop in self.site_constraints
        return satisfies_constraints