import string
from ..sage_helper import _within_sage, sage_method
def _normalize_exponents(self, exponents):
    D = self.elementary_divisors
    return [v % d if d > 0 else v for v, d in zip(exponents, D)]