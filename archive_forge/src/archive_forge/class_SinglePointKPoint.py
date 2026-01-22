import numpy as np
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.calculator import PropertyNotImplementedError
class SinglePointKPoint:

    def __init__(self, weight, s, k, eps_n=[], f_n=[]):
        self.weight = weight
        self.s = s
        self.k = k
        self.eps_n = eps_n
        self.f_n = f_n