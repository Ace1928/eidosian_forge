from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
class Deformation(SquareTensor):
    """Subclass of SquareTensor that describes the deformation gradient tensor."""
    symbol = 'd'

    def __new__(cls, deformation_gradient) -> Self:
        """
        Create a Deformation object. Note that the constructor uses __new__ rather than
        __init__ according to the standard method of subclassing numpy ndarrays.

        Args:
            deformation_gradient (3x3 array-like): the 3x3 array-like
                representing the deformation gradient
        """
        obj = super().__new__(cls, deformation_gradient)
        return obj.view(cls)

    def is_independent(self, tol: float=1e-08):
        """Checks to determine whether the deformation is independent."""
        return len(self.get_perturbed_indices(tol)) == 1

    def get_perturbed_indices(self, tol: float=1e-08):
        """
        Gets indices of perturbed elements of the deformation gradient,
        i. e. those that differ from the identity.
        """
        return list(zip(*np.where(abs(self - np.eye(3)) > tol)))

    @property
    def green_lagrange_strain(self):
        """Calculates the Euler-Lagrange strain from the deformation gradient."""
        return Strain.from_deformation(self)

    def apply_to_structure(self, structure: Structure):
        """
        Apply the deformation gradient to a structure.

        Args:
            structure (Structure object): the structure object to
                be modified by the deformation
        """
        def_struct = structure.copy()
        old_latt = def_struct.lattice.matrix
        new_latt = np.transpose(np.dot(self, np.transpose(old_latt)))
        def_struct.lattice = Lattice(new_latt)
        return def_struct

    @classmethod
    def from_index_amount(cls, matrix_pos, amt) -> Self:
        """
        Factory method for constructing a Deformation object
        from a matrix position and amount.

        Args:
            matrix_pos (tuple): tuple corresponding the matrix position to
                have a perturbation added
            amt (float): amount to add to the identity matrix at position
                matrix_pos
        """
        ident = np.identity(3)
        ident[matrix_pos] += amt
        return cls(ident)