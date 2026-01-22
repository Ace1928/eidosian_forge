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
class DopingTransformation(AbstractTransformation):
    """A transformation that performs doping of a structure."""

    def __init__(self, dopant, ionic_radius_tol=float('inf'), min_length=10, alio_tol=0, codopant=False, max_structures_per_enum=100, allowed_doping_species=None, **kwargs):
        """
        Args:
            dopant (Species-like): E.g., Al3+. Must have oxidation state.
            ionic_radius_tol (float): E.g., Fractional allowable ionic radii
                mismatch for dopant to fit into a site. Default of inf means
                that any dopant with the right oxidation state is allowed.
            min_length (float): Min. lattice parameter between periodic
                images of dopant. Defaults to 10A for now.
            alio_tol (int): If this is not 0, attempt will be made to dope
                sites with oxidation_states +- alio_tol of the dopant. E.g.,
                1 means that the ions like Ca2+ and Ti4+ are considered as
                potential doping sites for Al3+.
            codopant (bool): If True, doping will be carried out with a
                codopant to maintain charge neutrality. Otherwise, vacancies
                will be used.
            max_structures_per_enum (float): Maximum number of structures to
                return per enumeration. Note that there can be more than one
                candidate doping site, and each site enumeration will return at
                max max_structures_per_enum structures. Defaults to 100.
            allowed_doping_species (list): Species that are allowed to be
                doping sites. This is an inclusionary list. If specified,
                any sites which are not
            **kwargs:
                Same keyword args as EnumerateStructureTransformation,
                i.e., min_cell_size, etc.
        """
        self.dopant = get_el_sp(dopant)
        self.ionic_radius_tol = ionic_radius_tol
        self.min_length = min_length
        self.alio_tol = alio_tol
        self.codopant = codopant
        self.max_structures_per_enum = max_structures_per_enum
        self.allowed_doping_species = allowed_doping_species
        self.kwargs = kwargs

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False):
        """
        Args:
            structure (Structure): Input structure to dope
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures.
                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            list[dict] | Structure: each dict has shape {"structure": Structure, "energy": float}.
        """
        comp = structure.composition
        logger.info(f'Composition: {comp}')
        for sp in comp:
            try:
                sp.oxi_state
            except AttributeError:
                analyzer = BVAnalyzer()
                structure = analyzer.get_oxi_state_decorated_structure(structure)
                comp = structure.composition
                break
        ox = self.dopant.oxi_state
        radius = self.dopant.ionic_radius
        compatible_species = [sp for sp in comp if sp.oxi_state == ox and abs(sp.ionic_radius / radius - 1) < self.ionic_radius_tol]
        if not compatible_species and self.alio_tol:
            compatible_species = [sp for sp in comp if abs(sp.oxi_state - ox) <= self.alio_tol and abs(sp.ionic_radius / radius - 1) < self.ionic_radius_tol and (sp.oxi_state * ox >= 0)]
        if self.allowed_doping_species is not None:
            compatible_species = [sp for sp in compatible_species if sp in [get_el_sp(s) for s in self.allowed_doping_species]]
        logger.info(f'Compatible species: {compatible_species}')
        lengths = structure.lattice.abc
        scaling = [max(1, int(round(math.ceil(self.min_length / x)))) for x in lengths]
        logger.info(f'lengths={lengths!r}')
        logger.info(f'scaling={scaling!r}')
        all_structures: list[dict] = []
        trafo = EnumerateStructureTransformation(**self.kwargs)
        for sp in compatible_species:
            supercell = structure * scaling
            nsp = supercell.composition[sp]
            if sp.oxi_state == ox:
                supercell.replace_species({sp: {sp: (nsp - 1) / nsp, self.dopant: 1 / nsp}})
                logger.info(f'Doping {sp} for {self.dopant} at level {1 / nsp:.3f}')
            elif self.codopant:
                codopant = find_codopant(sp, 2 * sp.oxi_state - ox)
                supercell.replace_species({sp: {sp: (nsp - 2) / nsp, self.dopant: 1 / nsp, codopant: 1 / nsp}})
                logger.info(f'Doping {sp} for {self.dopant} + {codopant} at level {1 / nsp:.3f}')
            elif abs(sp.oxi_state) < abs(ox):
                sp_to_remove = min((s for s in comp if s.oxi_state * ox > 0), key=lambda ss: abs(ss.oxi_state))
                if sp_to_remove == sp:
                    common_charge = lcm(int(abs(sp.oxi_state)), int(abs(ox)))
                    n_dopant = common_charge / abs(ox)
                    nsp_to_remove = common_charge / abs(sp.oxi_state)
                    logger.info(f'Doping {nsp_to_remove} {sp} with {n_dopant} {self.dopant}.')
                    supercell.replace_species({sp: {sp: (nsp - nsp_to_remove) / nsp, self.dopant: n_dopant / nsp}})
                else:
                    ox_diff = int(abs(round(sp.oxi_state - ox)))
                    vac_ox = int(abs(sp_to_remove.oxi_state)) * ox_diff
                    common_charge = lcm(vac_ox, ox_diff)
                    n_dopant = common_charge / ox_diff
                    nx_to_remove = common_charge / vac_ox
                    nx = supercell.composition[sp_to_remove]
                    logger.info(f'Doping {n_dopant} {sp} with {self.dopant} and removing {nx_to_remove} {sp_to_remove}.')
                    supercell.replace_species({sp: {sp: (nsp - n_dopant) / nsp, self.dopant: n_dopant / nsp}, sp_to_remove: {sp_to_remove: (nx - nx_to_remove) / nx}})
            elif abs(sp.oxi_state) > abs(ox):
                if ox > 0:
                    sp_to_remove = max(supercell.composition, key=lambda el: el.X)
                else:
                    sp_to_remove = min(supercell.composition, key=lambda el: el.X)
                assert sp_to_remove.oxi_state * sp.oxi_state < 0
                ox_diff = int(abs(round(sp.oxi_state - ox)))
                anion_ox = int(abs(sp_to_remove.oxi_state))
                nx = supercell.composition[sp_to_remove]
                common_charge = lcm(anion_ox, ox_diff)
                n_dopant = common_charge / ox_diff
                nx_to_remove = common_charge / anion_ox
                logger.info(f'Doping {n_dopant} {sp} with {self.dopant} and removing {nx_to_remove} {sp_to_remove}.')
                supercell.replace_species({sp: {sp: (nsp - n_dopant) / nsp, self.dopant: n_dopant / nsp}, sp_to_remove: {sp_to_remove: (nx - nx_to_remove) / nx}})
            structs = trafo.apply_transformation(supercell, return_ranked_list=self.max_structures_per_enum)
            logger.info(f'{len(structs)} distinct structures')
            all_structures.extend(structs)
        logger.info(f'Total {len(all_structures)} doped structures')
        if return_ranked_list:
            return all_structures[:return_ranked_list]
        return all_structures[0]['structure']

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True