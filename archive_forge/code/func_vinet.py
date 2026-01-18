import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def vinet(V, E0, B0, BP, V0):
    """Vinet equation from PRB 70, 224107"""
    eta = (V / V0) ** (1 / 3)
    E = E0 + 2 * B0 * V0 / (BP - 1) ** 2 * (2 - (5 + 3 * BP * (eta - 1) - 3 * eta) * np.exp(-3 * (BP - 1) * (eta - 1) / 2))
    return E