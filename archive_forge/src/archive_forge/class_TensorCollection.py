from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class TensorCollection(collections.abc.Sequence, MSONable):
    """A sequence of tensors that can be used for fitting data
    or for having a tensor expansion.
    """

    def __init__(self, tensor_list: Sequence, base_class=Tensor) -> None:
        """
        Args:
            tensor_list: List of tensors.
            base_class: Class to be used.
        """
        self.tensors = [tensor if isinstance(tensor, base_class) else base_class(tensor) for tensor in tensor_list]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, ind):
        return self.tensors[ind]

    def __iter__(self):
        return iter(self.tensors)

    def zeroed(self, tol: float=0.001):
        """
        Args:
            tol: Tolerance.

        Returns:
            TensorCollection where small values are set to 0.
        """
        return type(self)([tensor.zeroed(tol) for tensor in self])

    def transform(self, symm_op):
        """Transforms TensorCollection with a symmetry operation.

        Args:
            symm_op: SymmetryOperation.

        Returns:
            TensorCollection.
        """
        return type(self)([tensor.transform(symm_op) for tensor in self])

    def rotate(self, matrix, tol: float=0.001):
        """Rotates TensorCollection.

        Args:
            matrix: Rotation matrix.
            tol: tolerance.

        Returns:
            TensorCollection.
        """
        return type(self)([tensor.rotate(matrix, tol) for tensor in self])

    @property
    def symmetrized(self):
        """TensorCollection where all tensors are symmetrized."""
        return type(self)([tensor.symmetrized for tensor in self])

    def is_symmetric(self, tol: float=1e-05) -> bool:
        """
        Args:
            tol: tolerance.

        Returns:
            Whether all tensors are symmetric.
        """
        return all((tensor.is_symmetric(tol) for tensor in self))

    def fit_to_structure(self, structure: Structure, symprec: float=0.1):
        """Fits all tensors to a Structure.

        Args:
            structure: Structure
            symprec: symmetry precision.

        Returns:
            TensorCollection.
        """
        return type(self)([tensor.fit_to_structure(structure, symprec) for tensor in self])

    def is_fit_to_structure(self, structure: Structure, tol: float=0.01):
        """
        Args:
            structure: Structure
            tol: tolerance.

        Returns:
            Whether all tensors are fitted to Structure.
        """
        return all((tensor.is_fit_to_structure(structure, tol) for tensor in self))

    @property
    def voigt(self):
        """TensorCollection where all tensors are in Voigt form."""
        return [tensor.voigt for tensor in self]

    @property
    def ranks(self):
        """Ranks for all tensors."""
        return [tensor.rank for tensor in self]

    def is_voigt_symmetric(self, tol: float=1e-06) -> bool:
        """
        Args:
            tol: tolerance.

        Returns:
            Whether all tensors are voigt symmetric.
        """
        return all((tensor.is_voigt_symmetric(tol) for tensor in self))

    @classmethod
    def from_voigt(cls, voigt_input_list, base_class=Tensor) -> Self:
        """Creates TensorCollection from voigt form.

        Args:
            voigt_input_list: List of voigt tensors
            base_class: Class for tensor.

        Returns:
            TensorCollection.
        """
        return cls([base_class.from_voigt(v) for v in voigt_input_list])

    def convert_to_ieee(self, structure: Structure, initial_fit=True, refine_rotation=True):
        """Convert all tensors to IEEE.

        Args:
            structure: Structure
            initial_fit: Whether to perform an initial fit.
            refine_rotation: Whether to refine the rotation.

        Returns:
            TensorCollection.
        """
        return type(self)([tensor.convert_to_ieee(structure, initial_fit, refine_rotation) for tensor in self])

    def round(self, *args, **kwargs):
        """Round all tensors.

        Args:
            args: Passthrough to Tensor.round
            kwargs: Passthrough to Tensor.round

        Returns:
            TensorCollection.
        """
        return type(self)([tensor.round(*args, **kwargs) for tensor in self])

    @property
    def voigt_symmetrized(self):
        """TensorCollection where all tensors are voigt symmetrized."""
        return type(self)([tensor.voigt_symmetrized for tensor in self])

    def as_dict(self, voigt=False):
        """
        Args:
            voigt: Whether to use Voigt form.

        Returns:
            Dict representation of TensorCollection.
        """
        tensor_list = self.voigt if voigt else self
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'tensor_list': [tensor.tolist() for tensor in tensor_list]}
        if voigt:
            dct['voigt'] = voigt
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Creates TensorCollection from dict.

        Args:
            dct: dict

        Returns:
            TensorCollection
        """
        voigt = dct.get('voigt')
        if voigt:
            return cls.from_voigt(dct['tensor_list'])
        return cls(dct['tensor_list'])