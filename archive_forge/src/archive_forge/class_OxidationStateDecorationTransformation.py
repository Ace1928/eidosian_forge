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
class OxidationStateDecorationTransformation(AbstractTransformation):
    """This transformation decorates a structure with oxidation states."""

    def __init__(self, oxidation_states):
        """
        Args:
            oxidation_states (dict): Oxidation states supplied as a dict,
            e.g., {"Li":1, "O":-2}.
        """
        self.oxidation_states = oxidation_states

    def apply_transformation(self, structure):
        """Apply the transformation.

        Args:
            structure (Structure): Input Structure

        Returns:
            Oxidation state decorated Structure.
        """
        struct = structure.copy()
        struct.add_oxidation_state_by_element(self.oxidation_states)
        return struct

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False