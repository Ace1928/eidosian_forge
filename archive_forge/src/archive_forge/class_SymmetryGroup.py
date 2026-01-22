from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, overload
import numpy as np
from monty.design_patterns import cached_class
from monty.serialization import loadfn
from pymatgen.util.string import Stringify
class SymmetryGroup(Sequence, Stringify, ABC):
    """Abstract class representing a symmetry group."""

    @property
    @abstractmethod
    def symmetry_ops(self) -> set[SymmOp]:
        """
        Returns:
            List of symmetry operations associated with the group.
        """

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, SymmOp):
            return NotImplemented
        return any((np.allclose(i.affine_matrix, item.affine_matrix) for i in self.symmetry_ops))

    def __hash__(self) -> int:
        return len(self)

    @overload
    def __getitem__(self, item: int) -> SymmOp:
        ...

    @overload
    def __getitem__(self, item: slice) -> Sequence[SymmOp]:
        ...

    def __getitem__(self, item: int | slice) -> SymmOp | Sequence[SymmOp]:
        return list(self.symmetry_ops)[item]

    def __len__(self) -> int:
        return len(self.symmetry_ops)

    def is_subgroup(self, supergroup: SymmetryGroup) -> bool:
        """True if this group is a subgroup of the supplied group.

        Args:
            supergroup (SymmetryGroup): Supergroup to test.

        Returns:
            bool: True if this group is a subgroup of the supplied group.
        """
        warnings.warn('This is not fully functional. Only trivial subsets are tested right now. ')
        return set(self.symmetry_ops).issubset(supergroup.symmetry_ops)

    def is_supergroup(self, subgroup: SymmetryGroup) -> bool:
        """True if this group is a supergroup of the supplied group.

        Args:
            subgroup (SymmetryGroup): Subgroup to test.

        Returns:
            bool: True if this group is a supergroup of the supplied group.
        """
        warnings.warn('This is not fully functional. Only trivial subsets are tested right now. ')
        return set(subgroup.symmetry_ops).issubset(self.symmetry_ops)

    def to_latex_string(self) -> str:
        """
        Returns:
            A latex formatted group symbol with proper subscripts and overlines.
        """
        sym = re.sub('_(\\d+)', '$_{\\1}$', self.to_pretty_string())
        return re.sub('-(\\d)', '$\\\\overline{\\1}$', sym)