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
class NthOrderElasticTensor(Tensor):
    """
    An object representing an nth-order tensor expansion
    of the stress-strain constitutive equations.
    """
    GPa_to_eV_A3 = Unit('GPa').get_conversion_factor(Unit('eV ang^-3'))
    symbol = 'C'

    def __new__(cls, input_array, check_rank=None, tol: float=0.0001) -> Self:
        """
        Args:
            input_array ():
            check_rank ():
            tol ():
        """
        obj = super().__new__(cls, input_array, check_rank=check_rank)
        if obj.rank % 2 != 0:
            raise ValueError('ElasticTensor must have even rank')
        if not obj.is_voigt_symmetric(tol):
            warnings.warn('Input elastic tensor does not satisfy standard Voigt symmetries')
        return obj.view(cls)

    @property
    def order(self):
        """Order of the elastic tensor."""
        return self.rank // 2

    def calculate_stress(self, strain):
        """
        Calculate's a given elastic tensor's contribution to the
        stress using Einstein summation.

        Args:
            strain (3x3 array-like): matrix corresponding to strain
        """
        strain = np.array(strain)
        if strain.shape == (6,):
            strain = Strain.from_voigt(strain)
        assert strain.shape == (3, 3), 'Strain must be 3x3 or Voigt notation'
        stress_matrix = self.einsum_sequence([strain] * (self.order - 1)) / factorial(self.order - 1)
        return Stress(stress_matrix)

    def energy_density(self, strain, convert_GPa_to_eV=True):
        """Calculates the elastic energy density due to a strain."""
        e_density = np.sum(self.calculate_stress(strain) * strain) / self.order
        if convert_GPa_to_eV:
            e_density *= self.GPa_to_eV_A3
        return e_density

    @classmethod
    def from_diff_fit(cls, strains, stresses, eq_stress=None, order=2, tol: float=1e-10) -> Self:
        """
        Takes a list of strains and stresses, and returns a list of coefficients for a
        polynomial fit of the given order.

        Args:
            strains: a list of strain values
            stresses: the stress values
            eq_stress: The stress at which the material is assumed to be elastic.
            order: The order of the polynomial to fit. Defaults to 2
            tol (float): tolerance for the fit.

        Returns:
            NthOrderElasticTensor: the fitted elastic tensor
        """
        return cls(diff_fit(strains, stresses, eq_stress, order, tol)[order - 2])