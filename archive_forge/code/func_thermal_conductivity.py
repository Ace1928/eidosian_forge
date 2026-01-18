from __future__ import annotations
import logging
from collections import defaultdict
import numpy as np
from monty.dev import deprecated
from scipy.constants import physical_constants
from scipy.integrate import quadrature
from scipy.misc import derivative
from scipy.optimize import minimize
from pymatgen.analysis.eos import EOS, PolynomialEOS
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
def thermal_conductivity(self, temperature: float, volume: float) -> float:
    """Eq(17) in 10.1103/PhysRevB.90.174107.

        Args:
            temperature (float): temperature in K
            volume (float): in Ang^3

        Returns:
            float: thermal conductivity in W/K/m
        """
    gamma = self.gruneisen_parameter(temperature, volume)
    theta_d = self.debye_temperature(volume)
    theta_a = theta_d * self.natoms ** (-1 / 3)
    prefactor = 0.849 * 3 * 4 ** (1 / 3) / (20.0 * np.pi ** 3)
    prefactor = prefactor * (self.kb / self.hbar) ** 3 * self.avg_mass
    kappa = prefactor / (gamma ** 2 - 0.514 * gamma + 0.228)
    return kappa * theta_a ** 2 * volume ** (1 / 3) * 1e-10