from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        Generate a symmetrized force constant matrix from an unsymmetrized matrix
        that has no unstable modes and also obeys the acoustic sum rule through an
        iterative procedure.

        Args:
            force (float): maximum force constant
            asum (int): number of iterations to attempt to obey the acoustic sum
                rule

        Returns:
            NxNx3x3 np.array representing the force constant matrix
        