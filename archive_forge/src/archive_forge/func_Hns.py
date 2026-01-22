import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@np.vectorize
def Hns(n, s):
    """Naive implementation of harmonic sum"""
    return (1 / np.arange(1, n + 1) ** s).sum()