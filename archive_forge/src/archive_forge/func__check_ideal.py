from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
def _check_ideal(self, J):
    """Helper to check ``J`` is an ideal of our ring."""
    if not isinstance(J, Ideal) or J.ring != self.ring:
        raise ValueError('J must be an ideal of %s, got %s' % (self.ring, J))