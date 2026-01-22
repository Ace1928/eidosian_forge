import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def antonschmidt(V, Einf, B, n, V0):
    """From Intermetallics 11, 23-32 (2003)

    Einf should be E_infinity, i.e. infinite separation, but
    according to the paper it does not provide a good estimate
    of the cohesive energy. They derive this equation from an
    empirical formula for the volume dependence of pressure,

    E(vol) = E_inf + int(P dV) from V=vol to V=infinity

    but the equation breaks down at large volumes, so E_inf
    is not that meaningful

    n should be about -2 according to the paper.

    I find this equation does not fit volumetric data as well
    as the other equtions do.
    """
    E = B * V0 / (n + 1) * (V / V0) ** (n + 1) * (np.log(V / V0) - 1 / (n + 1)) + Einf
    return E