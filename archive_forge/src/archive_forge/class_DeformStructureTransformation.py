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
class DeformStructureTransformation(AbstractTransformation):
    """This transformation deforms a structure by a deformation gradient matrix."""

    def __init__(self, deformation=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        """
        Args:
            deformation (array): deformation gradient for the transformation.
        """
        self._deform = Deformation(deformation)
        self.deformation = self._deform.tolist()

    def apply_transformation(self, structure):
        """Apply the transformation.

        Args:
            structure (Structure): Input Structure

        Returns:
            Deformed Structure.
        """
        return self._deform.apply_to_structure(structure)

    def __repr__(self):
        return f'DeformStructureTransformation : Deformation = {self.deformation}'

    @property
    def inverse(self):
        """Returns inverse Transformation."""
        return DeformStructureTransformation(self._deform.inv)

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False