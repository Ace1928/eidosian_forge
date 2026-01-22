from sympy.core.numbers import oo
from sympy.core.sympify import CantSympify
from sympy.polys.polyerrors import CoercionFailed, NotReversible, NotInvertible
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.sqfreetools import (
from sympy.polys.factortools import (
from sympy.polys.rootisolation import (
from sympy.polys.polyerrors import (
class GenericPoly(PicklableWithSlots):
    """Base class for low-level polynomial representations. """

    def ground_to_ring(f):
        """Make the ground domain a ring. """
        return f.set_domain(f.dom.get_ring())

    def ground_to_field(f):
        """Make the ground domain a field. """
        return f.set_domain(f.dom.get_field())

    def ground_to_exact(f):
        """Make the ground domain exact. """
        return f.set_domain(f.dom.get_exact())

    @classmethod
    def _perify_factors(per, result, include):
        if include:
            coeff, factors = result
        factors = [(per(g), k) for g, k in factors]
        if include:
            return (coeff, factors)
        else:
            return factors