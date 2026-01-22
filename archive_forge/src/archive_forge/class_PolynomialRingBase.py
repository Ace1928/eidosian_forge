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
class PolynomialRingBase(Ring, CharacteristicZero, CompositeDomain):
    """
    Base class for generalized polynomial rings.

    This base class should be used for uniform access to generalized polynomial
    rings. Subclasses only supply information about the element storage etc.

    Do not instantiate.
    """
    has_assoc_Ring = True
    has_assoc_Field = True
    default_order = 'grevlex'

    def __init__(self, dom, *gens, **opts):
        if not gens:
            raise GeneratorsNeeded('generators not specified')
        lev = len(gens) - 1
        self.ngens = len(gens)
        self.zero = self.dtype.zero(lev, dom, ring=self)
        self.one = self.dtype.one(lev, dom, ring=self)
        self.domain = self.dom = dom
        self.symbols = self.gens = gens
        self.order = opts.get('order', monomial_key(self.default_order))

    def new(self, element):
        return self.dtype(element, self.dom, len(self.gens) - 1, ring=self)

    def __str__(self):
        s_order = str(self.order)
        orderstr = ' order=' + s_order if s_order != self.default_order else ''
        return str(self.dom) + '[' + ','.join(map(str, self.gens)) + orderstr + ']'

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom, self.gens, self.order))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, PolynomialRingBase) and self.dtype == other.dtype and (self.dom == other.dom) and (self.gens == other.gens) and (self.order == other.order)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert a ``ANP`` object to ``dtype``. """
        if K1.dom == K0:
            return K1(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a ``PolyElement`` object to ``dtype``. """
        if K1.gens == K0.symbols:
            if K1.dom == K0.dom:
                return K1(dict(a))
            else:
                convert_dom = lambda c: K1.dom.convert_from(c, K0.dom)
                return K1({m: convert_dom(c) for m, c in a.items()})
        else:
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.symbols, K1.gens)
            if K1.dom != K0.dom:
                coeffs = [K1.dom.convert(c, K0.dom) for c in coeffs]
            return K1(dict(zip(monoms, coeffs)))

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert a ``DMP`` object to ``dtype``. """
        if K1.gens == K0.gens:
            if K1.dom == K0.dom:
                return K1(a.rep)
            else:
                return K1(a.convert(K1.dom).rep)
        else:
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.gens, K1.gens)
            if K1.dom != K0.dom:
                coeffs = [K1.dom.convert(c, K0.dom) for c in coeffs]
            return K1(dict(zip(monoms, coeffs)))

    def get_field(self):
        """Returns a field associated with ``self``. """
        return FractionField(self.dom, *self.gens)

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. ``K[X]``. """
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. ``K(X)``. """
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        try:
            return 1 / a
        except (ExactQuotientFailed, ZeroDivisionError):
            raise NotReversible('%s is not a unit' % a)

    def gcdex(self, a, b):
        """Extended GCD of ``a`` and ``b``. """
        return a.gcdex(b)

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return a.gcd(b)

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a.lcm(b)

    def factorial(self, a):
        """Returns factorial of ``a``. """
        return self.dtype(self.dom.factorial(a))

    def _vector_to_sdm(self, v, order):
        """
        For internal use by the modules class.

        Convert an iterable of elements of this ring into a sparse distributed
        module element.
        """
        raise NotImplementedError

    def _sdm_to_dics(self, s, n):
        """Helper for _sdm_to_vector."""
        from sympy.polys.distributedmodules import sdm_to_dict
        dic = sdm_to_dict(s)
        res = [{} for _ in range(n)]
        for k, v in dic.items():
            res[k[0]][k[1:]] = v
        return res

    def _sdm_to_vector(self, s, n):
        """
        For internal use by the modules class.

        Convert a sparse distributed module into a list of length ``n``.

        Examples
        ========

        >>> from sympy import QQ, ilex
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y, order=ilex)
        >>> L = [((1, 1, 1), QQ(1)), ((0, 1, 0), QQ(1)), ((0, 0, 1), QQ(2))]
        >>> R._sdm_to_vector(L, 2)
        [x + 2*y, x*y]
        """
        dics = self._sdm_to_dics(s, n)
        return [self(x) for x in dics]

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2)
        QQ[x]**2
        """
        return FreeModulePolyRing(self, rank)