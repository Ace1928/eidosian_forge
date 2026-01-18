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
@cite_gibbs
def vibrational_internal_energy(self, temperature, volume):
    """
        Vibrational internal energy, U_vib(V, T).
        Eq(4) in doi.org/10.1016/j.comphy.2003.12.001.

        Args:
            temperature (float): temperature in K
            volume (float): in Ang^3

        Returns:
            float: vibrational internal energy in eV
        """
    y = self.debye_temperature(volume) / temperature
    return self.kb * self.natoms * temperature * (9.0 / 8.0 * y + 3 * self.debye_integral(y))