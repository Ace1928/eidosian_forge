import numpy as np
from statsmodels.genmod.families import Poisson
from .gee_gaussian_simulation_check import GEE_simulator
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable,Independence
class Overdispersed_simulator(GEE_simulator):
    """
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
    """

    def print_dparams(self, dparams_est):
        OUT.write('Estimated inverse scale parameter:       %8.4f\n' % dparams_est[0])
        OUT.write('True inverse scale parameter:            %8.4f\n' % self.scale_inv)
        OUT.write('\n')

    def simulate(self):
        endog, exog, group, time = ([], [], [], [])
        f = np.sum(self.params ** 2)
        u, s, vt = np.linalg.svd(np.eye(len(self.params)) - np.outer(self.params, self.params) / f)
        params0 = u[:, np.flatnonzero(s > 1e-06)]
        for i in range(self.ngroups):
            gsize = np.random.randint(self.group_size_range[0], self.group_size_range[1])
            group.append([i] * gsize)
            time1 = np.random.normal(size=(gsize, 2))
            time.append(time1)
            exog1 = np.random.normal(size=(gsize, len(self.params)))
            exog.append(exog1)
            E = np.exp(np.dot(exog1, self.params))
            V = E * self.scale_inv
            p = E / V
            n = E * p / (1 - p)
            endog1 = np.random.negative_binomial(n, p, gsize)
            endog.append(endog1)
        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)