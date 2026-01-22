from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
class FranckCondonOverlap:
    """Evaluate squared overlaps depending on the Huang-Rhys parameter."""

    def __init__(self):
        self.factorial = Factorial()

    def directT0(self, n, S):
        """|<0|n>|^2

        Direct squared Franck-Condon overlap corresponding to T=0.
        """
        return np.exp(-S) * S ** n * self.factorial.inv(n)

    def direct(self, n, m, S_in):
        """|<n|m>|^2

        Direct squared Franck-Condon overlap.
        """
        if n > m:
            return self.direct(m, n, S_in)
        S = np.array([S_in])
        mask = np.where(S == 0)
        S[mask] = 1
        s = 0
        for k in range(n + 1):
            s += (-1) ** (n - k) * S ** float(-k) / (self.factorial(k) * self.factorial(n - k) * self.factorial(m - k))
        res = np.exp(-S) * S ** (n + m) * s ** 2 * (self.factorial(n) * self.factorial(m))
        res[mask] = int(n == m)
        return res[0]

    def direct0mm1(self, m, S):
        """<0|m><m|1>"""
        sum = S ** m
        if m:
            sum -= m * S ** (m - 1)
        return np.exp(-S) * np.sqrt(S) * sum * self.factorial.inv(m)

    def direct0mm2(self, m, S):
        """<0|m><m|2>"""
        sum = S ** (m + 1)
        if m >= 1:
            sum -= 2 * m * S ** m
        if m >= 2:
            sum += m * (m - 1) * S ** (m - 1)
        return np.exp(-S) / np.sqrt(2) * sum * self.factorial.inv(m)