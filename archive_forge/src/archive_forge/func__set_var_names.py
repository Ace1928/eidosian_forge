from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def _set_var_names(self, gens):
    if len(set(gens)) != len(gens) or not set(gens).issubset(string.ascii_lowercase):
        raise ValueError('Generators are unsuitable')
    poly_vars = list(gens) + list(combinations(gens, 2))
    poly_vars += list(combinations(gens, 3))
    self.var_names = ['T' + ''.join(v) for v in poly_vars]