from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass
class _PageL:
    """Maintains state between `page_trend_test` executions"""

    def __init__(self):
        """Lightweight initialization"""
        self.all_pmfs = {}

    def set_k(self, k):
        """Calculate lower and upper limits of L for single row"""
        self.k = k
        self.a, self.b = (k * (k + 1) * (k + 2) // 6, k * (k + 1) * (2 * k + 1) // 6)

    def sf(self, l, n):
        """Survival function of Page's L statistic"""
        ps = [self.pmf(l, n) for l in range(l, n * self.b + 1)]
        return np.sum(ps)

    def p_l_k_1(self):
        """Relative frequency of each L value over all possible single rows"""
        ranks = range(1, self.k + 1)
        rank_perms = np.array(list(permutations(ranks)))
        Ls = (ranks * rank_perms).sum(axis=1)
        counts = np.histogram(Ls, np.arange(self.a - 0.5, self.b + 1.5))[0]
        return counts / math.factorial(self.k)

    def pmf(self, l, n):
        """Recursive function to evaluate p(l, k, n); see [5] Equation 1"""
        if n not in self.all_pmfs:
            self.all_pmfs[n] = {}
        if self.k not in self.all_pmfs[n]:
            self.all_pmfs[n][self.k] = {}
        if l in self.all_pmfs[n][self.k]:
            return self.all_pmfs[n][self.k][l]
        if n == 1:
            ps = self.p_l_k_1()
            ls = range(self.a, self.b + 1)
            self.all_pmfs[n][self.k] = {l: p for l, p in zip(ls, ps)}
            return self.all_pmfs[n][self.k][l]
        p = 0
        low = max(l - (n - 1) * self.b, self.a)
        high = min(l - (n - 1) * self.a, self.b)
        for t in range(low, high + 1):
            p1 = self.pmf(l - t, n - 1)
            p2 = self.pmf(t, 1)
            p += p1 * p2
        self.all_pmfs[n][self.k][l] = p
        return p