import numpy as np
from statsmodels.genmod.families import Poisson
from .gee_gaussian_simulation_check import GEE_simulator
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable,Independence

    Use the negative binomial distribution to check GEE estimation
    using the overdispered Poisson model with independent dependence.

    Simulating
        X = np.random.negative_binomial(n, p, size)
    then EX = (1 - p) * n / p
         Var(X) = (1 - p) * n / p**2

    These equations can be inverted as follows:

        p = E / V
        n = E * p / (1 - p)

    dparams[0] is the common correlation coefficient
    