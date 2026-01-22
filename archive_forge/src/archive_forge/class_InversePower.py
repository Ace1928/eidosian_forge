import numpy as np
import scipy.stats
import warnings
class InversePower(Power):
    """
    The inverse transform

    Notes
    -----
    g(p) = 1/p

    Alias of statsmodels.family.links.Power(power=-1.)
    """

    def __init__(self):
        super().__init__(power=-1.0)