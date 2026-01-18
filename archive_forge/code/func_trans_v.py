from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
@raise_if_unphysical
def trans_v(self, structure: Structure) -> float:
    """
        Calculates transverse sound velocity using the
        Voigt-Reuss-Hill average bulk modulus.

        Args:
            structure: pymatgen structure object

        Returns:
            float: transverse sound velocity (in SI units)
        """
    n_sites = len(structure)
    n_atoms = structure.composition.num_atoms
    weight = float(structure.composition.weight)
    mass_density = 1660.5 * n_sites * weight / (n_atoms * structure.volume)
    if self.g_vrh < 0:
        raise ValueError('k_vrh or g_vrh is negative, sound velocity is undefined')
    return (1000000000.0 * self.g_vrh / mass_density) ** 0.5