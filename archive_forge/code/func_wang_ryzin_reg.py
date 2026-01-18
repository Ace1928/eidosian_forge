import numpy as np
from scipy.special import erf
def wang_ryzin_reg(h, Xi, x):
    """
    A version for the Wang-Ryzin kernel for nonparametric regression.

    Suggested by Li and Racine in [1] ch.4
    """
    return h ** abs(Xi - x)