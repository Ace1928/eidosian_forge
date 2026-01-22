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
class RotationTransformation(AbstractTransformation):
    """The RotationTransformation applies a rotation to a structure."""

    def __init__(self, axis, angle, angle_in_radians=False):
        """
        Args:
            axis (3x1 array): Axis of rotation, e.g., [1, 0, 0]
            angle (float): Angle to rotate
            angle_in_radians (bool): Set to True if angle is supplied in radians.
                Else degrees are assumed.
        """
        self.axis = axis
        self.angle = angle
        self.angle_in_radians = angle_in_radians
        self._symmop = SymmOp.from_axis_angle_and_translation(self.axis, self.angle, self.angle_in_radians)

    def apply_transformation(self, structure):
        """Apply the transformation.

        Args:
            structure (Structure): Input Structure

        Returns:
            Rotated Structure.
        """
        struct = structure.copy()
        struct.apply_operation(self._symmop)
        return struct

    def __repr__(self):
        return f'Rotation Transformation about axis {self.axis} with angle = {self.angle:.4f} {('radians' if self.angle_in_radians else 'degrees')}'

    @property
    def inverse(self):
        """Returns inverse Transformation."""
        return RotationTransformation(self.axis, -self.angle, self.angle_in_radians)

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False