from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import (GeneratorsNeeded, PolynomialError,
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable
@public
class GlobalPolynomialRing(PolynomialRingBase):
    """A true polynomial ring, with objects DMP. """
    is_PolynomialRing = is_Poly = True
    dtype = DMP

    def from_FractionField(K1, a, K0):
        """
        Convert a ``DMF`` object to ``DMP``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP, DMF
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.abc import x

        >>> f = DMF(([ZZ(1), ZZ(1)], [ZZ(1)]), ZZ)
        >>> K = ZZ.old_frac_field(x)

        >>> F = ZZ.old_poly_ring(x).from_FractionField(f, K)

        >>> F == DMP([ZZ(1), ZZ(1)], ZZ)
        True
        >>> type(F)
        <class 'sympy.polys.polyclasses.DMP'>

        """
        if a.denom().is_one:
            return K1.from_GlobalPolynomialRing(a.numer(), K0)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return basic_from_dict(a.to_sympy_dict(), *self.gens)

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        try:
            rep, _ = dict_from_basic(a, gens=self.gens)
        except PolynomialError:
            raise CoercionFailed('Cannot convert %s to type %s' % (a, self))
        for k, v in rep.items():
            rep[k] = self.dom.from_sympy(v)
        return self(rep)

    def is_positive(self, a):
        """Returns True if ``LC(a)`` is positive. """
        return self.dom.is_positive(a.LC())

    def is_negative(self, a):
        """Returns True if ``LC(a)`` is negative. """
        return self.dom.is_negative(a.LC())

    def is_nonpositive(self, a):
        """Returns True if ``LC(a)`` is non-positive. """
        return self.dom.is_nonpositive(a.LC())

    def is_nonnegative(self, a):
        """Returns True if ``LC(a)`` is non-negative. """
        return self.dom.is_nonnegative(a.LC())

    def _vector_to_sdm(self, v, order):
        """
        Examples
        ========

        >>> from sympy import lex, QQ
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y)
        >>> f = R.convert(x + 2*y)
        >>> g = R.convert(x * y)
        >>> R._vector_to_sdm([f, g], lex)
        [((1, 1, 1), 1), ((0, 1, 0), 1), ((0, 0, 1), 2)]
        """
        return _vector_to_sdm_helper(v, order)