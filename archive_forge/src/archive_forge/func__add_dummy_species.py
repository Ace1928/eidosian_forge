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
def _add_dummy_species(structure, order_parameters):
    """
        Args:
            structure: ordered Structure
            order_parameters: list of MagOrderParameterConstraints.

        Returns:
            A structure decorated with disordered
            DummySpecies on which to perform the enumeration.
            Note that the DummySpecies are super-imposed on
            to the original sites, to make it easier to
            retrieve the original site after enumeration is
            performed (this approach is preferred over a simple
            mapping since multiple species may have the same
            DummySpecies, depending on the constraints specified).
            This approach can also preserve site properties even after
            enumeration.
        """
    dummy_struct = structure.copy()

    def generate_dummy_specie():
        """Generator which returns DummySpecies symbols Mma, Mmb, etc."""
        subscript_length = 1
        while True:
            for subscript in product(ascii_lowercase, repeat=subscript_length):
                yield ('Mm' + ''.join(subscript))
            subscript_length += 1
    dummy_species_gen = generate_dummy_specie()
    dummy_species_symbols = [next(dummy_species_gen) for i in range(len(order_parameters))]
    dummy_species = [{DummySpecies(symbol, spin=Spin.up): constraint.order_parameter, DummySpecies(symbol, spin=Spin.down): 1 - constraint.order_parameter} for symbol, constraint in zip(dummy_species_symbols, order_parameters)]
    for site in dummy_struct:
        satisfies_constraints = [c.satisfies_constraint(site) for c in order_parameters]
        if satisfies_constraints.count(True) > 1:
            raise ValueError(f'Order parameter constraints conflict for site: {site.specie}, {site.properties}')
        if any(satisfies_constraints):
            dummy_specie_idx = satisfies_constraints.index(True)
            dummy_struct.append(dummy_species[dummy_specie_idx], site.coords, site.lattice)
    return dummy_struct