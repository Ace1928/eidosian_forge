from __future__ import annotations
import logging
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from numpy import around
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, get_el_sp
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.site_transformations import PartialRemoveSitesTransformation
from pymatgen.transformations.transformation_abc import AbstractTransformation
class SubstitutionTransformation(AbstractTransformation):
    """This transformation substitutes species for one another."""

    def __init__(self, species_map: dict[SpeciesLike, SpeciesLike | dict[SpeciesLike, float]] | list[tuple[SpeciesLike, SpeciesLike]]) -> None:
        """
        Args:
            species_map: A dict or list of tuples containing the species mapping in
                string-string pairs. E.g., {"Li": "Na"} or [("Fe2+","Mn2+")].
                Multiple substitutions can be done. Overloaded to accept
                sp_and_occu dictionary E.g. {"Si: {"Ge":0.75, "C":0.25}},
                which substitutes a single species with multiple species to
                generate a disordered structure.
        """
        self.species_map = species_map
        self._species_map = dict(species_map)
        for k, v in self._species_map.items():
            if isinstance(v, (tuple, list)):
                self._species_map[k] = dict(v)

    def apply_transformation(self, structure: Structure) -> Structure:
        """Apply the transformation.

        Args:
            structure (Structure): Input Structure

        Returns:
            Substituted Structure.
        """
        species_map = {}
        for k, v in self._species_map.items():
            value = {get_el_sp(x): y for x, y in v.items()} if isinstance(v, dict) else get_el_sp(v)
            species_map[get_el_sp(k)] = value
        struct = structure.copy()
        struct.replace_species(species_map)
        return struct

    def __repr__(self):
        return 'Substitution Transformation :' + ', '.join([f'{k}->{v}' for k, v in self._species_map.items()])

    @property
    def inverse(self):
        """Returns inverse Transformation."""
        inverse_map = {v: k for k, v in self._species_map.items()}
        return SubstitutionTransformation(inverse_map)

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False