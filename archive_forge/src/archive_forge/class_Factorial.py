from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
class Factorial:

    def __init__(self):
        self._fac = [1]
        self._inv = [1.0]

    def __call__(self, n):
        try:
            return self._fac[n]
        except IndexError:
            for i in range(len(self._fac), n + 1):
                self._fac.append(i * self._fac[i - 1])
                try:
                    self._inv.append(float(1.0 / self._fac[-1]))
                except OverflowError:
                    self._inv.append(0.0)
            return self._fac[n]

    def inv(self, n):
        self(n)
        return self._inv[n]