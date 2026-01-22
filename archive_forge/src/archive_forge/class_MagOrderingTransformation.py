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
class MagOrderingTransformation(AbstractTransformation):
    """This transformation takes a structure and returns a list of collinear
    magnetic orderings. For disordered structures, make an ordered
    approximation first.
    """

    def __init__(self, mag_species_spin, order_parameter=0.5, energy_model=None, **kwargs):
        """
        Args:
            mag_species_spin: A mapping of elements/species to their
                spin magnitudes, e.g. {"Fe3+": 5, "Mn3+": 4}
            order_parameter (float or list): if float, a specifies a
                global order parameter and can take values from 0.0 to 1.0
                (e.g. 0.5 for antiferromagnetic or 1.0 for ferromagnetic), if
                list has to be a list of
                pymatgen.transformations.advanced_transformations.MagOrderParameterConstraint
                to specify more complicated orderings, see documentation for
                MagOrderParameterConstraint more details on usage
            energy_model: Energy model to rank the returned structures,
                see :mod: `pymatgen.analysis.energy_models` for more information (note
                that this is not necessarily a physical energy). By default, returned
                structures use SymmetryModel() which ranks structures from most
                symmetric to least.
            kwargs: Additional kwargs that are passed to
                EnumerateStructureTransformation such as min_cell_size etc.
        """
        if isinstance(order_parameter, float):
            order_parameter = [MagOrderParameterConstraint(order_parameter=order_parameter, species_constraints=list(mag_species_spin))]
        elif isinstance(order_parameter, list):
            ops = [isinstance(item, MagOrderParameterConstraint) for item in order_parameter]
            if not any(ops):
                raise ValueError('Order parameter not correctly defined.')
        else:
            raise ValueError('Order parameter not correctly defined.')
        self.mag_species_spin = mag_species_spin
        self.order_parameter = [op.as_dict() for op in order_parameter]
        self.energy_model = energy_model or SymmetryModel()
        self.enum_kwargs = kwargs

    @staticmethod
    def determine_min_cell(disordered_structure):
        """Determine the smallest supercell that is able to enumerate
        the provided structure with the given order parameter.
        """

        def lcm(n1, n2):
            """Find least common multiple of two numbers."""
            return n1 * n2 / gcd(n1, n2)
        mag_species_order_parameter = {}
        mag_species_occurrences = {}
        for site in disordered_structure:
            if not site.is_ordered:
                sp = str(next(iter(site.species))).split(',', maxsplit=1)[0]
                if sp in mag_species_order_parameter:
                    mag_species_occurrences[sp] += 1
                else:
                    op = max(site.species.values())
                    mag_species_order_parameter[sp] = op
                    mag_species_occurrences[sp] = 1
        smallest_n = []
        for sp, order_parameter in mag_species_order_parameter.items():
            denom = Fraction(order_parameter).limit_denominator(100).denominator
            num_atom_per_specie = mag_species_occurrences[sp]
            n_gcd = gcd(denom, num_atom_per_specie)
            smallest_n.append(lcm(int(n_gcd), denom) / n_gcd)
        return max(smallest_n)

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

    @staticmethod
    def _remove_dummy_species(structure):
        """Structure with dummy species removed, but their corresponding spin properties
        merged with the original sites. Used after performing enumeration.
        """
        if not structure.is_ordered:
            raise RuntimeError('Something went wrong with enumeration.')
        sites_to_remove = []
        logger.debug(f'Dummy species structure:\n{structure}')
        for idx, site in enumerate(structure):
            if isinstance(site.specie, DummySpecies):
                sites_to_remove.append(idx)
                spin = getattr(site.specie, 'spin', 0)
                neighbors = structure.get_neighbors(site, 0.05, include_index=True)
                if len(neighbors) != 1:
                    raise RuntimeError(f"This shouldn't happen, found neighbors={neighbors!r}")
                orig_site_idx = neighbors[0][2]
                orig_specie = structure[orig_site_idx].specie
                new_specie = Species(orig_specie.symbol, getattr(orig_specie, 'oxi_state', None), spin=spin)
                structure.replace(orig_site_idx, new_specie, properties=structure[orig_site_idx].properties)
        structure.remove_sites(sites_to_remove)
        logger.debug(f'Structure with dummy species removed:\n{structure}')
        return structure

    def _add_spin_magnitudes(self, structure: Structure):
        """Replaces Spin.up/Spin.down with spin magnitudes specified by mag_species_spin.

        Args:
            structure (Structure): Structure to modify.

        Returns:
            Structure: Structure with spin magnitudes added.
        """
        for idx, site in enumerate(structure):
            if getattr(site.specie, 'spin', None):
                spin = site.specie.spin
                spin = getattr(site.specie, 'spin', None)
                sign = int(spin) if spin else 0
                if spin:
                    sp = str(site.specie).split(',', maxsplit=1)[0]
                    new_spin = sign * self.mag_species_spin.get(sp, 0)
                    new_specie = Species(site.specie.symbol, getattr(site.specie, 'oxi_state', None), spin=new_spin)
                    structure.replace(idx, new_specie, properties=site.properties)
        logger.debug(f'Structure with spin magnitudes:\n{structure}')
        return structure

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False) -> Structure | list[Structure]:
        """Apply MagOrderTransformation to an input structure.

        Args:
            structure (Structure): Any ordered structure.
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures
                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Raises:
            ValueError: On disordered structures.

        Returns:
            Structure | list[Structure]: Structure(s) after MagOrderTransformation.
        """
        if not structure.is_ordered:
            raise ValueError('Create an ordered approximation of your  input structure first.')
        order_parameters = [MagOrderParameterConstraint.from_dict(op_dict) for op_dict in self.order_parameter]
        structure = self._add_dummy_species(structure, order_parameters)
        if structure.is_ordered:
            structure = self._remove_dummy_species(structure)
            return [structure] if return_ranked_list > 1 else structure
        enum_kwargs = self.enum_kwargs.copy()
        enum_kwargs['min_cell_size'] = max(int(self.determine_min_cell(structure)), enum_kwargs.get('min_cell_size', 1))
        if enum_kwargs.get('max_cell_size'):
            if enum_kwargs['min_cell_size'] > enum_kwargs['max_cell_size']:
                warnings.warn(f'Specified max cell size ({enum_kwargs['max_cell_size']}) is smaller than the minimum enumerable cell size ({enum_kwargs['min_cell_size']}), changing max cell size to {enum_kwargs['min_cell_size']}')
                enum_kwargs['max_cell_size'] = enum_kwargs['min_cell_size']
        else:
            enum_kwargs['max_cell_size'] = enum_kwargs['min_cell_size']
        trafo = EnumerateStructureTransformation(**enum_kwargs)
        alls = trafo.apply_transformation(structure, return_ranked_list=return_ranked_list)
        if isinstance(alls, Structure):
            alls = self._remove_dummy_species(alls)
            alls = self._add_spin_magnitudes(alls)
        else:
            for idx in range(len(alls)):
                alls[idx]['structure'] = self._remove_dummy_species(alls[idx]['structure'])
                alls[idx]['structure'] = self._add_spin_magnitudes(alls[idx]['structure'])
        try:
            num_to_return = int(return_ranked_list)
        except ValueError:
            num_to_return = 1
        if num_to_return == 1 or not return_ranked_list:
            return alls[0]['structure'] if num_to_return else alls
        matcher = StructureMatcher(comparator=SpinComparator())

        def key(struct: Structure) -> int:
            return SpacegroupAnalyzer(struct, 0.1).get_space_group_number()
        out = []
        for _, group in groupby(sorted((dct['structure'] for dct in alls), key=key), key):
            group = list(group)
            grouped = matcher.group_structures(group)
            out.extend([{'structure': g[0], 'energy': self.energy_model.get_energy(g[0])} for g in grouped])
        self._all_structures = sorted(out, key=lambda dct: dct['energy'])
        return self._all_structures[0:num_to_return]

    def __str__(self) -> str:
        return 'MagOrderingTransformation'

    def __repr__(self) -> str:
        return str(self)

    @property
    def inverse(self) -> None:
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True