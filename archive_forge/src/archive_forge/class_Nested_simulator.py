from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested
class Nested_simulator(GEE_simulator):
    nest_sizes = None
    id_matrix = None

    def print_dparams(self, dparams_est):
        for j in range(len(self.nest_sizes)):
            OUT.write('Nest %d variance estimate:  %8.4f\n' % (j + 1, dparams_est[j]))
            OUT.write('Nest %d variance truth:     %8.4f\n' % (j + 1, self.dparams[j]))
        OUT.write('Error variance estimate:   %8.4f\n' % (dparams_est[-1] - sum(dparams_est[0:-1])))
        OUT.write('Error variance truth:      %8.4f\n' % self.error_sd ** 2)
        OUT.write('\n')

    def simulate(self):
        group_effect_var = self.dparams[0]
        vcomp = self.dparams[1:]
        vcomp.append(0)
        endog, exog, group, id_matrix = ([], [], [], [])
        for i in range(self.ngroups):
            iterators = [lrange(n) for n in self.nest_sizes]
            variances = [np.sqrt(v) * np.random.normal(size=n) for v, n in zip(vcomp, self.nest_sizes)]
            gpe = np.random.normal() * np.sqrt(group_effect_var)
            nest_all = []
            for j in self.nest_sizes:
                nest_all.append(set())
            for nest in product(*iterators):
                group.append(i)
                ref = gpe + sum([v[j] for v, j in zip(variances, nest)])
                exog1 = np.random.normal(size=5)
                exog1[0] = 1
                exog.append(exog1)
                error = ref + self.error_sd * np.random.normal()
                endog1 = np.dot(exog1, self.params) + error
                endog.append(endog1)
                for j in range(len(nest)):
                    nest_all[j].add(tuple(nest[0:j + 1]))
                nest1 = [len(x) - 1 for x in nest_all]
                id_matrix.append(nest1[0:-1])
        self.exog = np.array(exog)
        self.endog = np.array(endog)
        self.group = np.array(group)
        self.id_matrix = np.array(id_matrix)
        self.time = np.zeros_like(self.endog)