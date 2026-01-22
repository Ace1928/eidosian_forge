from __future__ import annotations
from typing import Any
from operator import add, mul, lt, le, gt, ge
from functools import reduce
from types import GeneratorType
from sympy.core.expr import Expr
from sympy.core.numbers import igcd, oo
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify, sympify
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.compatibility import IPolys
from sympy.polys.constructor import construct_domain
from sympy.polys.densebasic import dmp_to_dict, dmp_from_dict
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import (
from sympy.polys.polyoptions import (Domain as DomainOpt,
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public, subsets
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):
    """Element of multivariate distributed polynomial ring. """

    def new(self, init):
        return self.__class__(init)

    def parent(self):
        return self.ring.to_domain()

    def __getnewargs__(self):
        return (self.ring, list(self.iterterms()))
    _hash = None

    def __hash__(self):
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.ring, frozenset(self.items())))
        return _hash

    def copy(self):
        """Return a copy of polynomial self.

        Polynomials are mutable; if one is interested in preserving
        a polynomial, and one plans to use inplace operations, one
        can copy the polynomial. This method makes a shallow copy.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> R, x, y = ring('x, y', ZZ)
        >>> p = (x + y)**2
        >>> p1 = p.copy()
        >>> p2 = p
        >>> p[R.zero_monom] = 3
        >>> p
        x**2 + 2*x*y + y**2 + 3
        >>> p1
        x**2 + 2*x*y + y**2
        >>> p2
        x**2 + 2*x*y + y**2 + 3

        """
        return self.new(self)

    def set_ring(self, new_ring):
        if self.ring == new_ring:
            return self
        elif self.ring.symbols != new_ring.symbols:
            terms = list(zip(*_dict_reorder(self, self.ring.symbols, new_ring.symbols)))
            return new_ring.from_terms(terms, self.ring.domain)
        else:
            return new_ring.from_dict(self, self.ring.domain)

    def as_expr(self, *symbols):
        if not symbols:
            symbols = self.ring.symbols
        elif len(symbols) != self.ring.ngens:
            raise ValueError('Wrong number of symbols, expected %s got %s' % (self.ring.ngens, len(symbols)))
        return expr_from_dict(self.as_expr_dict(), *symbols)

    def as_expr_dict(self):
        to_sympy = self.ring.domain.to_sympy
        return {monom: to_sympy(coeff) for monom, coeff in self.iterterms()}

    def clear_denoms(self):
        domain = self.ring.domain
        if not domain.is_Field or not domain.has_assoc_Ring:
            return (domain.one, self)
        ground_ring = domain.get_ring()
        common = ground_ring.one
        lcm = ground_ring.lcm
        denom = domain.denom
        for coeff in self.values():
            common = lcm(common, denom(coeff))
        poly = self.new([(k, v * common) for k, v in self.items()])
        return (common, poly)

    def strip_zero(self):
        """Eliminate monomials with zero coefficient. """
        for k, v in list(self.items()):
            if not v:
                del self[k]

    def __eq__(p1, p2):
        """Equality test for polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = (x + y)**2 + (x - y)**2
        >>> p1 == 4*x*y
        False
        >>> p1 == 2*(x**2 + y**2)
        True

        """
        if not p2:
            return not p1
        elif isinstance(p2, PolyElement) and p2.ring == p1.ring:
            return dict.__eq__(p1, p2)
        elif len(p1) > 1:
            return False
        else:
            return p1.get(p1.ring.zero_monom) == p2

    def __ne__(p1, p2):
        return not p1 == p2

    def almosteq(p1, p2, tolerance=None):
        """Approximate equality test for polynomials. """
        ring = p1.ring
        if isinstance(p2, ring.dtype):
            if set(p1.keys()) != set(p2.keys()):
                return False
            almosteq = ring.domain.almosteq
            for k in p1.keys():
                if not almosteq(p1[k], p2[k], tolerance):
                    return False
            return True
        elif len(p1) > 1:
            return False
        else:
            try:
                p2 = ring.domain.convert(p2)
            except CoercionFailed:
                return False
            else:
                return ring.domain.almosteq(p1.const(), p2, tolerance)

    def sort_key(self):
        return (len(self), self.terms())

    def _cmp(p1, p2, op):
        if isinstance(p2, p1.ring.dtype):
            return op(p1.sort_key(), p2.sort_key())
        else:
            return NotImplemented

    def __lt__(p1, p2):
        return p1._cmp(p2, lt)

    def __le__(p1, p2):
        return p1._cmp(p2, le)

    def __gt__(p1, p2):
        return p1._cmp(p2, gt)

    def __ge__(p1, p2):
        return p1._cmp(p2, ge)

    def _drop(self, gen):
        ring = self.ring
        i = ring.index(gen)
        if ring.ngens == 1:
            return (i, ring.domain)
        else:
            symbols = list(ring.symbols)
            del symbols[i]
            return (i, ring.clone(symbols=symbols))

    def drop(self, gen):
        i, ring = self._drop(gen)
        if self.ring.ngens == 1:
            if self.is_ground:
                return self.coeff(1)
            else:
                raise ValueError('Cannot drop %s' % gen)
        else:
            poly = ring.zero
            for k, v in self.items():
                if k[i] == 0:
                    K = list(k)
                    del K[i]
                    poly[tuple(K)] = v
                else:
                    raise ValueError('Cannot drop %s' % gen)
            return poly

    def _drop_to_ground(self, gen):
        ring = self.ring
        i = ring.index(gen)
        symbols = list(ring.symbols)
        del symbols[i]
        return (i, ring.clone(symbols=symbols, domain=ring[i]))

    def drop_to_ground(self, gen):
        if self.ring.ngens == 1:
            raise ValueError('Cannot drop only generator to ground')
        i, ring = self._drop_to_ground(gen)
        poly = ring.zero
        gen = ring.domain.gens[0]
        for monom, coeff in self.iterterms():
            mon = monom[:i] + monom[i + 1:]
            if mon not in poly:
                poly[mon] = (gen ** monom[i]).mul_ground(coeff)
            else:
                poly[mon] += (gen ** monom[i]).mul_ground(coeff)
        return poly

    def to_dense(self):
        return dmp_from_dict(self, self.ring.ngens - 1, self.ring.domain)

    def to_dict(self):
        return dict(self)

    def str(self, printer, precedence, exp_pattern, mul_symbol):
        if not self:
            return printer._print(self.ring.domain.zero)
        prec_mul = precedence['Mul']
        prec_atom = precedence['Atom']
        ring = self.ring
        symbols = ring.symbols
        ngens = ring.ngens
        zm = ring.zero_monom
        sexpvs = []
        for expv, coeff in self.terms():
            negative = ring.domain.is_negative(coeff)
            sign = ' - ' if negative else ' + '
            sexpvs.append(sign)
            if expv == zm:
                scoeff = printer._print(coeff)
                if negative and scoeff.startswith('-'):
                    scoeff = scoeff[1:]
            else:
                if negative:
                    coeff = -coeff
                if coeff != self.ring.domain.one:
                    scoeff = printer.parenthesize(coeff, prec_mul, strict=True)
                else:
                    scoeff = ''
            sexpv = []
            for i in range(ngens):
                exp = expv[i]
                if not exp:
                    continue
                symbol = printer.parenthesize(symbols[i], prec_atom, strict=True)
                if exp != 1:
                    if exp != int(exp) or exp < 0:
                        sexp = printer.parenthesize(exp, prec_atom, strict=False)
                    else:
                        sexp = exp
                    sexpv.append(exp_pattern % (symbol, sexp))
                else:
                    sexpv.append('%s' % symbol)
            if scoeff:
                sexpv = [scoeff] + sexpv
            sexpvs.append(mul_symbol.join(sexpv))
        if sexpvs[0] in [' + ', ' - ']:
            head = sexpvs.pop(0)
            if head == ' - ':
                sexpvs.insert(0, '-')
        return ''.join(sexpvs)

    @property
    def is_generator(self):
        return self in self.ring._gens_set

    @property
    def is_ground(self):
        return not self or (len(self) == 1 and self.ring.zero_monom in self)

    @property
    def is_monomial(self):
        return not self or (len(self) == 1 and self.LC == 1)

    @property
    def is_term(self):
        return len(self) <= 1

    @property
    def is_negative(self):
        return self.ring.domain.is_negative(self.LC)

    @property
    def is_positive(self):
        return self.ring.domain.is_positive(self.LC)

    @property
    def is_nonnegative(self):
        return self.ring.domain.is_nonnegative(self.LC)

    @property
    def is_nonpositive(self):
        return self.ring.domain.is_nonpositive(self.LC)

    @property
    def is_zero(f):
        return not f

    @property
    def is_one(f):
        return f == f.ring.one

    @property
    def is_monic(f):
        return f.ring.domain.is_one(f.LC)

    @property
    def is_primitive(f):
        return f.ring.domain.is_one(f.content())

    @property
    def is_linear(f):
        return all((sum(monom) <= 1 for monom in f.itermonoms()))

    @property
    def is_quadratic(f):
        return all((sum(monom) <= 2 for monom in f.itermonoms()))

    @property
    def is_squarefree(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_sqf_p(f)

    @property
    def is_irreducible(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_irreducible_p(f)

    @property
    def is_cyclotomic(f):
        if f.ring.is_univariate:
            return f.ring.dup_cyclotomic_p(f)
        else:
            raise MultivariatePolynomialError('cyclotomic polynomial')

    def __neg__(self):
        return self.new([(monom, -coeff) for monom, coeff in self.iterterms()])

    def __pos__(self):
        return self

    def __add__(p1, p2):
        """Add two polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> (x + y)**2 + (x - y)**2
        2*x**2 + 2*y**2

        """
        if not p2:
            return p1.copy()
        ring = p1.ring
        if isinstance(p2, ring.dtype):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for k, v in p2.items():
                v = get(k, zero) + v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__radd__(p1)
            else:
                return NotImplemented
        try:
            cp2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            if not cp2:
                return p
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = cp2
            elif p2 == -p[zm]:
                del p[zm]
            else:
                p[zm] += cp2
            return p

    def __radd__(p1, n):
        p = p1.copy()
        if not n:
            return p
        ring = p1.ring
        try:
            n = ring.domain_new(n)
        except CoercionFailed:
            return NotImplemented
        else:
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = n
            elif n == -p[zm]:
                del p[zm]
            else:
                p[zm] += n
            return p

    def __sub__(p1, p2):
        """Subtract polynomial p2 from p1.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = x + y**2
        >>> p2 = x*y + y**2
        >>> p1 - p2
        -x*y + x

        """
        if not p2:
            return p1.copy()
        ring = p1.ring
        if isinstance(p2, ring.dtype):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for k, v in p2.items():
                v = get(k, zero) - v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rsub__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = -p2
            elif p2 == p[zm]:
                del p[zm]
            else:
                p[zm] -= p2
            return p

    def __rsub__(p1, n):
        """n - p1 with n convertible to the coefficient domain.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y
        >>> 4 - p
        -x - y + 4

        """
        ring = p1.ring
        try:
            n = ring.domain_new(n)
        except CoercionFailed:
            return NotImplemented
        else:
            p = ring.zero
            for expv in p1:
                p[expv] = -p1[expv]
            p += n
            return p

    def __mul__(p1, p2):
        """Multiply two polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', QQ)
        >>> p1 = x + y
        >>> p2 = x - y
        >>> p1*p2
        x**2 - y**2

        """
        ring = p1.ring
        p = ring.zero
        if not p1 or not p2:
            return p
        elif isinstance(p2, ring.dtype):
            get = p.get
            zero = ring.domain.zero
            monomial_mul = ring.monomial_mul
            p2it = list(p2.items())
            for exp1, v1 in p1.items():
                for exp2, v2 in p2it:
                    exp = monomial_mul(exp1, exp2)
                    p[exp] = get(exp, zero) + v1 * v2
            p.strip_zero()
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmul__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            for exp1, v1 in p1.items():
                v = v1 * p2
                if v:
                    p[exp1] = v
            return p

    def __rmul__(p1, p2):
        """p2 * p1 with p2 in the coefficient domain of p1.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y
        >>> 4 * p
        4*x + 4*y

        """
        p = p1.ring.zero
        if not p2:
            return p
        try:
            p2 = p.ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            for exp1, v1 in p1.items():
                v = p2 * v1
                if v:
                    p[exp1] = v
            return p

    def __pow__(self, n):
        """raise polynomial to power `n`

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p**3
        x**3 + 3*x**2*y**2 + 3*x*y**4 + y**6

        """
        ring = self.ring
        if not n:
            if self:
                return ring.one
            else:
                raise ValueError('0**0')
        elif len(self) == 1:
            monom, coeff = list(self.items())[0]
            p = ring.zero
            if coeff == ring.domain.one:
                p[ring.monomial_pow(monom, n)] = coeff
            else:
                p[ring.monomial_pow(monom, n)] = coeff ** n
            return p
        n = int(n)
        if n < 0:
            raise ValueError('Negative exponent')
        elif n == 1:
            return self.copy()
        elif n == 2:
            return self.square()
        elif n == 3:
            return self * self.square()
        elif len(self) <= 5:
            return self._pow_multinomial(n)
        else:
            return self._pow_generic(n)

    def _pow_generic(self, n):
        p = self.ring.one
        c = self
        while True:
            if n & 1:
                p = p * c
                n -= 1
                if not n:
                    break
            c = c.square()
            n = n // 2
        return p

    def _pow_multinomial(self, n):
        multinomials = multinomial_coefficients(len(self), n).items()
        monomial_mulpow = self.ring.monomial_mulpow
        zero_monom = self.ring.zero_monom
        terms = self.items()
        zero = self.ring.domain.zero
        poly = self.ring.zero
        for multinomial, multinomial_coeff in multinomials:
            product_monom = zero_monom
            product_coeff = multinomial_coeff
            for exp, (monom, coeff) in zip(multinomial, terms):
                if exp:
                    product_monom = monomial_mulpow(product_monom, monom, exp)
                    product_coeff *= coeff ** exp
            monom = tuple(product_monom)
            coeff = product_coeff
            coeff = poly.get(monom, zero) + coeff
            if coeff:
                poly[monom] = coeff
            elif monom in poly:
                del poly[monom]
        return poly

    def square(self):
        """square of a polynomial

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p.square()
        x**2 + 2*x*y**2 + y**4

        """
        ring = self.ring
        p = ring.zero
        get = p.get
        keys = list(self.keys())
        zero = ring.domain.zero
        monomial_mul = ring.monomial_mul
        for i in range(len(keys)):
            k1 = keys[i]
            pk = self[k1]
            for j in range(i):
                k2 = keys[j]
                exp = monomial_mul(k1, k2)
                p[exp] = get(exp, zero) + pk * self[k2]
        p = p.imul_num(2)
        get = p.get
        for k, v in self.items():
            k2 = monomial_mul(k, k)
            p[k2] = get(k2, zero) + v ** 2
        p.strip_zero()
        return p

    def __divmod__(p1, p2):
        ring = p1.ring
        if not p2:
            raise ZeroDivisionError('polynomial division')
        elif isinstance(p2, ring.dtype):
            return p1.div(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rdivmod__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return (p1.quo_ground(p2), p1.rem_ground(p2))

    def __rdivmod__(p1, p2):
        return NotImplemented

    def __mod__(p1, p2):
        ring = p1.ring
        if not p2:
            raise ZeroDivisionError('polynomial division')
        elif isinstance(p2, ring.dtype):
            return p1.rem(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmod__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.rem_ground(p2)

    def __rmod__(p1, p2):
        return NotImplemented

    def __truediv__(p1, p2):
        ring = p1.ring
        if not p2:
            raise ZeroDivisionError('polynomial division')
        elif isinstance(p2, ring.dtype):
            if p2.is_monomial:
                return p1 * p2 ** (-1)
            else:
                return p1.quo(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rtruediv__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.quo_ground(p2)

    def __rtruediv__(p1, p2):
        return NotImplemented
    __floordiv__ = __truediv__
    __rfloordiv__ = __rtruediv__

    def _term_div(self):
        zm = self.ring.zero_monom
        domain = self.ring.domain
        domain_quo = domain.quo
        monomial_div = self.ring.monomial_div
        if domain.is_Field:

            def term_div(a_lm_a_lc, b_lm_b_lc):
                a_lm, a_lc = a_lm_a_lc
                b_lm, b_lc = b_lm_b_lc
                if b_lm == zm:
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if monom is not None:
                    return (monom, domain_quo(a_lc, b_lc))
                else:
                    return None
        else:

            def term_div(a_lm_a_lc, b_lm_b_lc):
                a_lm, a_lc = a_lm_a_lc
                b_lm, b_lc = b_lm_b_lc
                if b_lm == zm:
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if not (monom is None or a_lc % b_lc):
                    return (monom, domain_quo(a_lc, b_lc))
                else:
                    return None
        return term_div

    def div(self, fv):
        """Division algorithm, see [CLO] p64.

        fv array of polynomials
           return qv, r such that
           self = sum(fv[i]*qv[i]) + r

        All polynomials are required not to be Laurent polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> f = x**3
        >>> f0 = x - y**2
        >>> f1 = x - y
        >>> qv, r = f.div((f0, f1))
        >>> qv[0]
        x**2 + x*y**2 + y**4
        >>> qv[1]
        0
        >>> r
        y**6

        """
        ring = self.ring
        ret_single = False
        if isinstance(fv, PolyElement):
            ret_single = True
            fv = [fv]
        if not all(fv):
            raise ZeroDivisionError('polynomial division')
        if not self:
            if ret_single:
                return (ring.zero, ring.zero)
            else:
                return ([], ring.zero)
        for f in fv:
            if f.ring != ring:
                raise ValueError('self and f must have the same ring')
        s = len(fv)
        qv = [ring.zero for i in range(s)]
        p = self.copy()
        r = ring.zero
        term_div = self._term_div()
        expvs = [fx.leading_expv() for fx in fv]
        while p:
            i = 0
            divoccurred = 0
            while i < s and divoccurred == 0:
                expv = p.leading_expv()
                term = term_div((expv, p[expv]), (expvs[i], fv[i][expvs[i]]))
                if term is not None:
                    expv1, c = term
                    qv[i] = qv[i]._iadd_monom((expv1, c))
                    p = p._iadd_poly_monom(fv[i], (expv1, -c))
                    divoccurred = 1
                else:
                    i += 1
            if not divoccurred:
                expv = p.leading_expv()
                r = r._iadd_monom((expv, p[expv]))
                del p[expv]
        if expv == ring.zero_monom:
            r += p
        if ret_single:
            if not qv:
                return (ring.zero, r)
            else:
                return (qv[0], r)
        else:
            return (qv, r)

    def rem(self, G):
        f = self
        if isinstance(G, PolyElement):
            G = [G]
        if not all(G):
            raise ZeroDivisionError('polynomial division')
        ring = f.ring
        domain = ring.domain
        zero = domain.zero
        monomial_mul = ring.monomial_mul
        r = ring.zero
        term_div = f._term_div()
        ltf = f.LT
        f = f.copy()
        get = f.get
        while f:
            for g in G:
                tq = term_div(ltf, g.LT)
                if tq is not None:
                    m, c = tq
                    for mg, cg in g.iterterms():
                        m1 = monomial_mul(mg, m)
                        c1 = get(m1, zero) - c * cg
                        if not c1:
                            del f[m1]
                        else:
                            f[m1] = c1
                    ltm = f.leading_expv()
                    if ltm is not None:
                        ltf = (ltm, f[ltm])
                    break
            else:
                ltm, ltc = ltf
                if ltm in r:
                    r[ltm] += ltc
                else:
                    r[ltm] = ltc
                del f[ltm]
                ltm = f.leading_expv()
                if ltm is not None:
                    ltf = (ltm, f[ltm])
        return r

    def quo(f, G):
        return f.div(G)[0]

    def exquo(f, G):
        q, r = f.div(G)
        if not r:
            return q
        else:
            raise ExactQuotientFailed(f, G)

    def _iadd_monom(self, mc):
        """add to self the monomial coeff*x0**i0*x1**i1*...
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x**4 + 2*y
        >>> m = (1, 2)
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        x**4 + 5*x*y**2 + 2*y
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        5*x*y**2 + x
        >>> p1 is p
        False

        """
        if self in self.ring._gens_set:
            cpself = self.copy()
        else:
            cpself = self
        expv, coeff = mc
        c = cpself.get(expv)
        if c is None:
            cpself[expv] = coeff
        else:
            c += coeff
            if c:
                cpself[expv] = c
            else:
                del cpself[expv]
        return cpself

    def _iadd_poly_monom(self, p2, mc):
        """add to self the product of (p)*(coeff*x0**i0*x1**i1*...)
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p1 = x**4 + 2*y
        >>> p2 = y + z
        >>> m = (1, 2, 3)
        >>> p1 = p1._iadd_poly_monom(p2, (m, 3))
        >>> p1
        x**4 + 3*x*y**3*z**3 + 3*x*y**2*z**4 + 2*y

        """
        p1 = self
        if p1 in p1.ring._gens_set:
            p1 = p1.copy()
        m, c = mc
        get = p1.get
        zero = p1.ring.domain.zero
        monomial_mul = p1.ring.monomial_mul
        for k, v in p2.items():
            ka = monomial_mul(k, m)
            coeff = get(ka, zero) + v * c
            if coeff:
                p1[ka] = coeff
            else:
                del p1[ka]
        return p1

    def degree(f, x=None):
        """
        The leading degree in ``x`` or the main variable.

        Note that the degree of 0 is negative infinity (the SymPy object -oo).

        """
        i = f.ring.index(x)
        if not f:
            return -oo
        elif i < 0:
            return 0
        else:
            return max([monom[i] for monom in f.itermonoms()])

    def degrees(f):
        """
        A tuple containing leading degrees in all variables.

        Note that the degree of 0 is negative infinity (the SymPy object -oo)

        """
        if not f:
            return (-oo,) * f.ring.ngens
        else:
            return tuple(map(max, list(zip(*f.itermonoms()))))

    def tail_degree(f, x=None):
        """
        The tail degree in ``x`` or the main variable.

        Note that the degree of 0 is negative infinity (the SymPy object -oo)

        """
        i = f.ring.index(x)
        if not f:
            return -oo
        elif i < 0:
            return 0
        else:
            return min([monom[i] for monom in f.itermonoms()])

    def tail_degrees(f):
        """
        A tuple containing tail degrees in all variables.

        Note that the degree of 0 is negative infinity (the SymPy object -oo)

        """
        if not f:
            return (-oo,) * f.ring.ngens
        else:
            return tuple(map(min, list(zip(*f.itermonoms()))))

    def leading_expv(self):
        """Leading monomial tuple according to the monomial ordering.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p = x**4 + x**3*y + x**2*z**2 + z**7
        >>> p.leading_expv()
        (4, 0, 0)

        """
        if self:
            return self.ring.leading_expv(self)
        else:
            return None

    def _get_coeff(self, expv):
        return self.get(expv, self.ring.domain.zero)

    def coeff(self, element):
        """
        Returns the coefficient that stands next to the given monomial.

        Parameters
        ==========

        element : PolyElement (with ``is_monomial = True``) or 1

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring("x,y,z", ZZ)
        >>> f = 3*x**2*y - x*y*z + 7*z**3 + 23

        >>> f.coeff(x**2*y)
        3
        >>> f.coeff(x*y)
        0
        >>> f.coeff(1)
        23

        """
        if element == 1:
            return self._get_coeff(self.ring.zero_monom)
        elif isinstance(element, self.ring.dtype):
            terms = list(element.iterterms())
            if len(terms) == 1:
                monom, coeff = terms[0]
                if coeff == self.ring.domain.one:
                    return self._get_coeff(monom)
        raise ValueError('expected a monomial, got %s' % element)

    def const(self):
        """Returns the constant coefficient. """
        return self._get_coeff(self.ring.zero_monom)

    @property
    def LC(self):
        return self._get_coeff(self.leading_expv())

    @property
    def LM(self):
        expv = self.leading_expv()
        if expv is None:
            return self.ring.zero_monom
        else:
            return expv

    def leading_monom(self):
        """
        Leading monomial as a polynomial element.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> (3*x*y + y**2).leading_monom()
        x*y

        """
        p = self.ring.zero
        expv = self.leading_expv()
        if expv:
            p[expv] = self.ring.domain.one
        return p

    @property
    def LT(self):
        expv = self.leading_expv()
        if expv is None:
            return (self.ring.zero_monom, self.ring.domain.zero)
        else:
            return (expv, self._get_coeff(expv))

    def leading_term(self):
        """Leading term as a polynomial element.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> (3*x*y + y**2).leading_term()
        3*x*y

        """
        p = self.ring.zero
        expv = self.leading_expv()
        if expv is not None:
            p[expv] = self[expv]
        return p

    def _sorted(self, seq, order):
        if order is None:
            order = self.ring.order
        else:
            order = OrderOpt.preprocess(order)
        if order is lex:
            return sorted(seq, key=lambda monom: monom[0], reverse=True)
        else:
            return sorted(seq, key=lambda monom: order(monom[0]), reverse=True)

    def coeffs(self, order=None):
        """Ordered list of polynomial coefficients.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.coeffs()
        [2, 1]
        >>> f.coeffs(grlex)
        [1, 2]

        """
        return [coeff for _, coeff in self.terms(order)]

    def monoms(self, order=None):
        """Ordered list of polynomial monomials.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.monoms()
        [(2, 3), (1, 7)]
        >>> f.monoms(grlex)
        [(1, 7), (2, 3)]

        """
        return [monom for monom, _ in self.terms(order)]

    def terms(self, order=None):
        """Ordered list of polynomial terms.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.terms()
        [((2, 3), 2), ((1, 7), 1)]
        >>> f.terms(grlex)
        [((1, 7), 1), ((2, 3), 2)]

        """
        return self._sorted(list(self.items()), order)

    def itercoeffs(self):
        """Iterator over coefficients of a polynomial. """
        return iter(self.values())

    def itermonoms(self):
        """Iterator over monomials of a polynomial. """
        return iter(self.keys())

    def iterterms(self):
        """Iterator over terms of a polynomial. """
        return iter(self.items())

    def listcoeffs(self):
        """Unordered list of polynomial coefficients. """
        return list(self.values())

    def listmonoms(self):
        """Unordered list of polynomial monomials. """
        return list(self.keys())

    def listterms(self):
        """Unordered list of polynomial terms. """
        return list(self.items())

    def imul_num(p, c):
        """multiply inplace the polynomial p by an element in the
        coefficient ring, provided p is not one of the generators;
        else multiply not inplace

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p1 = p.imul_num(3)
        >>> p1
        3*x + 3*y**2
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p.imul_num(3)
        >>> p1
        3*x
        >>> p1 is p
        False

        """
        if p in p.ring._gens_set:
            return p * c
        if not c:
            p.clear()
            return
        for exp in p:
            p[exp] *= c
        return p

    def content(f):
        """Returns GCD of polynomial's coefficients. """
        domain = f.ring.domain
        cont = domain.zero
        gcd = domain.gcd
        for coeff in f.itercoeffs():
            cont = gcd(cont, coeff)
        return cont

    def primitive(f):
        """Returns content and a primitive polynomial. """
        cont = f.content()
        return (cont, f.quo_ground(cont))

    def monic(f):
        """Divides all coefficients by the leading coefficient. """
        if not f:
            return f
        else:
            return f.quo_ground(f.LC)

    def mul_ground(f, x):
        if not x:
            return f.ring.zero
        terms = [(monom, coeff * x) for monom, coeff in f.iterterms()]
        return f.new(terms)

    def mul_monom(f, monom):
        monomial_mul = f.ring.monomial_mul
        terms = [(monomial_mul(f_monom, monom), f_coeff) for f_monom, f_coeff in f.items()]
        return f.new(terms)

    def mul_term(f, term):
        monom, coeff = term
        if not f or not coeff:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.mul_ground(coeff)
        monomial_mul = f.ring.monomial_mul
        terms = [(monomial_mul(f_monom, monom), f_coeff * coeff) for f_monom, f_coeff in f.items()]
        return f.new(terms)

    def quo_ground(f, x):
        domain = f.ring.domain
        if not x:
            raise ZeroDivisionError('polynomial division')
        if not f or x == domain.one:
            return f
        if domain.is_Field:
            quo = domain.quo
            terms = [(monom, quo(coeff, x)) for monom, coeff in f.iterterms()]
        else:
            terms = [(monom, coeff // x) for monom, coeff in f.iterterms() if not coeff % x]
        return f.new(terms)

    def quo_term(f, term):
        monom, coeff = term
        if not coeff:
            raise ZeroDivisionError('polynomial division')
        elif not f:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.quo_ground(coeff)
        term_div = f._term_div()
        terms = [term_div(t, term) for t in f.iterterms()]
        return f.new([t for t in terms if t is not None])

    def trunc_ground(f, p):
        if f.ring.domain.is_ZZ:
            terms = []
            for monom, coeff in f.iterterms():
                coeff = coeff % p
                if coeff > p // 2:
                    coeff = coeff - p
                terms.append((monom, coeff))
        else:
            terms = [(monom, coeff % p) for monom, coeff in f.iterterms()]
        poly = f.new(terms)
        poly.strip_zero()
        return poly
    rem_ground = trunc_ground

    def extract_ground(self, g):
        f = self
        fc = f.content()
        gc = g.content()
        gcd = f.ring.domain.gcd(fc, gc)
        f = f.quo_ground(gcd)
        g = g.quo_ground(gcd)
        return (gcd, f, g)

    def _norm(f, norm_func):
        if not f:
            return f.ring.domain.zero
        else:
            ground_abs = f.ring.domain.abs
            return norm_func([ground_abs(coeff) for coeff in f.itercoeffs()])

    def max_norm(f):
        return f._norm(max)

    def l1_norm(f):
        return f._norm(sum)

    def deflate(f, *G):
        ring = f.ring
        polys = [f] + list(G)
        J = [0] * ring.ngens
        for p in polys:
            for monom in p.itermonoms():
                for i, m in enumerate(monom):
                    J[i] = igcd(J[i], m)
        for i, b in enumerate(J):
            if not b:
                J[i] = 1
        J = tuple(J)
        if all((b == 1 for b in J)):
            return (J, polys)
        H = []
        for p in polys:
            h = ring.zero
            for I, coeff in p.iterterms():
                N = [i // j for i, j in zip(I, J)]
                h[tuple(N)] = coeff
            H.append(h)
        return (J, H)

    def inflate(f, J):
        poly = f.ring.zero
        for I, coeff in f.iterterms():
            N = [i * j for i, j in zip(I, J)]
            poly[tuple(N)] = coeff
        return poly

    def lcm(self, g):
        f = self
        domain = f.ring.domain
        if not domain.is_Field:
            fc, f = f.primitive()
            gc, g = g.primitive()
            c = domain.lcm(fc, gc)
        h = (f * g).quo(f.gcd(g))
        if not domain.is_Field:
            return h.mul_ground(c)
        else:
            return h.monic()

    def gcd(f, g):
        return f.cofactors(g)[0]

    def cofactors(f, g):
        if not f and (not g):
            zero = f.ring.zero
            return (zero, zero, zero)
        elif not f:
            h, cff, cfg = f._gcd_zero(g)
            return (h, cff, cfg)
        elif not g:
            h, cfg, cff = g._gcd_zero(f)
            return (h, cff, cfg)
        elif len(f) == 1:
            h, cff, cfg = f._gcd_monom(g)
            return (h, cff, cfg)
        elif len(g) == 1:
            h, cfg, cff = g._gcd_monom(f)
            return (h, cff, cfg)
        J, (f, g) = f.deflate(g)
        h, cff, cfg = f._gcd(g)
        return (h.inflate(J), cff.inflate(J), cfg.inflate(J))

    def _gcd_zero(f, g):
        one, zero = (f.ring.one, f.ring.zero)
        if g.is_nonnegative:
            return (g, zero, one)
        else:
            return (-g, zero, -one)

    def _gcd_monom(f, g):
        ring = f.ring
        ground_gcd = ring.domain.gcd
        ground_quo = ring.domain.quo
        monomial_gcd = ring.monomial_gcd
        monomial_ldiv = ring.monomial_ldiv
        mf, cf = list(f.iterterms())[0]
        _mgcd, _cgcd = (mf, cf)
        for mg, cg in g.iterterms():
            _mgcd = monomial_gcd(_mgcd, mg)
            _cgcd = ground_gcd(_cgcd, cg)
        h = f.new([(_mgcd, _cgcd)])
        cff = f.new([(monomial_ldiv(mf, _mgcd), ground_quo(cf, _cgcd))])
        cfg = f.new([(monomial_ldiv(mg, _mgcd), ground_quo(cg, _cgcd)) for mg, cg in g.iterterms()])
        return (h, cff, cfg)

    def _gcd(f, g):
        ring = f.ring
        if ring.domain.is_QQ:
            return f._gcd_QQ(g)
        elif ring.domain.is_ZZ:
            return f._gcd_ZZ(g)
        else:
            return ring.dmp_inner_gcd(f, g)

    def _gcd_ZZ(f, g):
        return heugcd(f, g)

    def _gcd_QQ(self, g):
        f = self
        ring = f.ring
        new_ring = ring.clone(domain=ring.domain.get_ring())
        cf, f = f.clear_denoms()
        cg, g = g.clear_denoms()
        f = f.set_ring(new_ring)
        g = g.set_ring(new_ring)
        h, cff, cfg = f._gcd_ZZ(g)
        h = h.set_ring(ring)
        c, h = (h.LC, h.monic())
        cff = cff.set_ring(ring).mul_ground(ring.domain.quo(c, cf))
        cfg = cfg.set_ring(ring).mul_ground(ring.domain.quo(c, cg))
        return (h, cff, cfg)

    def cancel(self, g):
        """
        Cancel common factors in a rational function ``f/g``.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> (2*x**2 - 2).cancel(x**2 - 2*x + 1)
        (2*x + 2, x - 1)

        """
        f = self
        ring = f.ring
        if not f:
            return (f, ring.one)
        domain = ring.domain
        if not (domain.is_Field and domain.has_assoc_Ring):
            _, p, q = f.cofactors(g)
        else:
            new_ring = ring.clone(domain=domain.get_ring())
            cq, f = f.clear_denoms()
            cp, g = g.clear_denoms()
            f = f.set_ring(new_ring)
            g = g.set_ring(new_ring)
            _, p, q = f.cofactors(g)
            _, cp, cq = new_ring.domain.cofactors(cp, cq)
            p = p.set_ring(ring)
            q = q.set_ring(ring)
            p = p.mul_ground(cp)
            q = q.mul_ground(cq)
        u = q.canonical_unit()
        if u == domain.one:
            p, q = (p, q)
        elif u == -domain.one:
            p, q = (-p, -q)
        else:
            p = p.mul_ground(u)
            q = q.mul_ground(u)
        return (p, q)

    def canonical_unit(f):
        domain = f.ring.domain
        return domain.canonical_unit(f.LC)

    def diff(f, x):
        """Computes partial derivative in ``x``.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring("x,y", ZZ)
        >>> p = x + x**2*y**3
        >>> p.diff(x)
        2*x*y**3 + 1

        """
        ring = f.ring
        i = ring.index(x)
        m = ring.monomial_basis(i)
        g = ring.zero
        for expv, coeff in f.iterterms():
            if expv[i]:
                e = ring.monomial_ldiv(expv, m)
                g[e] = ring.domain_new(coeff * expv[i])
        return g

    def __call__(f, *values):
        if 0 < len(values) <= f.ring.ngens:
            return f.evaluate(list(zip(f.ring.gens, values)))
        else:
            raise ValueError('expected at least 1 and at most %s values, got %s' % (f.ring.ngens, len(values)))

    def evaluate(self, x, a=None):
        f = self
        if isinstance(x, list) and a is None:
            (X, a), x = (x[0], x[1:])
            f = f.evaluate(X, a)
            if not x:
                return f
            else:
                x = [(Y.drop(X), a) for Y, a in x]
                return f.evaluate(x)
        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)
        if ring.ngens == 1:
            result = ring.domain.zero
            for (n,), coeff in f.iterterms():
                result += coeff * a ** n
            return result
        else:
            poly = ring.drop(x).zero
            for monom, coeff in f.iterterms():
                n, monom = (monom[i], monom[:i] + monom[i + 1:])
                coeff = coeff * a ** n
                if monom in poly:
                    coeff = coeff + poly[monom]
                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                elif coeff:
                    poly[monom] = coeff
            return poly

    def subs(self, x, a=None):
        f = self
        if isinstance(x, list) and a is None:
            for X, a in x:
                f = f.subs(X, a)
            return f
        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)
        if ring.ngens == 1:
            result = ring.domain.zero
            for (n,), coeff in f.iterterms():
                result += coeff * a ** n
            return ring.ground_new(result)
        else:
            poly = ring.zero
            for monom, coeff in f.iterterms():
                n, monom = (monom[i], monom[:i] + (0,) + monom[i + 1:])
                coeff = coeff * a ** n
                if monom in poly:
                    coeff = coeff + poly[monom]
                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                elif coeff:
                    poly[monom] = coeff
            return poly

    def symmetrize(self):
        """
        Rewrite *self* in terms of elementary symmetric polynomials.

        Explanation
        ===========

        If this :py:class:`~.PolyElement` belongs to a ring of $n$ variables,
        we can try to write it as a function of the elementary symmetric
        polynomials on $n$ variables. We compute a symmetric part, and a
        remainder for any part we were not able to symmetrize.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> R, x, y = ring("x,y", ZZ)

        >>> f = x**2 + y**2
        >>> f.symmetrize()
        (x**2 - 2*y, 0, [(x, x + y), (y, x*y)])

        >>> f = x**2 - y**2
        >>> f.symmetrize()
        (x**2 - 2*y, -2*y**2, [(x, x + y), (y, x*y)])

        Returns
        =======

        Triple ``(p, r, m)``
            ``p`` is a :py:class:`~.PolyElement` that represents our attempt
            to express *self* as a function of elementary symmetric
            polynomials. Each variable in ``p`` stands for one of the
            elementary symmetric polynomials. The correspondence is given
            by ``m``.

            ``r`` is the remainder.

            ``m`` is a list of pairs, giving the mapping from variables in
            ``p`` to elementary symmetric polynomials.

            The triple satisfies the equation ``p.compose(m) + r == self``.
            If the remainder ``r`` is zero, *self* is symmetric. If it is
            nonzero, we were not able to represent *self* as symmetric.

        See Also
        ========

        sympy.polys.polyfuncs.symmetrize

        References
        ==========

        .. [1] Lauer, E. Algorithms for symmetrical polynomials, Proc. 1976
            ACM Symp. on Symbolic and Algebraic Computing, NY 242-247.
            https://dl.acm.org/doi/pdf/10.1145/800205.806342

        """
        f = self.copy()
        ring = f.ring
        n = ring.ngens
        if not n:
            return (f, ring.zero, [])
        polys = [ring.symmetric_poly(i + 1) for i in range(n)]
        poly_powers = {}

        def get_poly_power(i, n):
            if (i, n) not in poly_powers:
                poly_powers[i, n] = polys[i] ** n
            return poly_powers[i, n]
        indices = list(range(n - 1))
        weights = list(range(n, 0, -1))
        symmetric = ring.zero
        while f:
            _height, _monom, _coeff = (-1, None, None)
            for i, (monom, coeff) in enumerate(f.terms()):
                if all((monom[i] >= monom[i + 1] for i in indices)):
                    height = max([n * m for n, m in zip(weights, monom)])
                    if height > _height:
                        _height, _monom, _coeff = (height, monom, coeff)
            if _height != -1:
                monom, coeff = (_monom, _coeff)
            else:
                break
            exponents = []
            for m1, m2 in zip(monom, monom[1:] + (0,)):
                exponents.append(m1 - m2)
            symmetric += ring.term_new(tuple(exponents), coeff)
            product = coeff
            for i, n in enumerate(exponents):
                product *= get_poly_power(i, n)
            f -= product
        mapping = list(zip(ring.gens, polys))
        return (symmetric, f, mapping)

    def compose(f, x, a=None):
        ring = f.ring
        poly = ring.zero
        gens_map = dict(zip(ring.gens, range(ring.ngens)))
        if a is not None:
            replacements = [(x, a)]
        elif isinstance(x, list):
            replacements = list(x)
        elif isinstance(x, dict):
            replacements = sorted(x.items(), key=lambda k: gens_map[k[0]])
        else:
            raise ValueError('expected a generator, value pair a sequence of such pairs')
        for k, (x, g) in enumerate(replacements):
            replacements[k] = (gens_map[x], ring.ring_new(g))
        for monom, coeff in f.iterterms():
            monom = list(monom)
            subpoly = ring.one
            for i, g in replacements:
                n, monom[i] = (monom[i], 0)
                if n:
                    subpoly *= g ** n
            subpoly = subpoly.mul_term((tuple(monom), coeff))
            poly += subpoly
        return poly

    def pdiv(f, g):
        return f.ring.dmp_pdiv(f, g)

    def prem(f, g):
        return f.ring.dmp_prem(f, g)

    def pquo(f, g):
        return f.ring.dmp_quo(f, g)

    def pexquo(f, g):
        return f.ring.dmp_exquo(f, g)

    def half_gcdex(f, g):
        return f.ring.dmp_half_gcdex(f, g)

    def gcdex(f, g):
        return f.ring.dmp_gcdex(f, g)

    def subresultants(f, g):
        return f.ring.dmp_subresultants(f, g)

    def resultant(f, g):
        return f.ring.dmp_resultant(f, g)

    def discriminant(f):
        return f.ring.dmp_discriminant(f)

    def decompose(f):
        if f.ring.is_univariate:
            return f.ring.dup_decompose(f)
        else:
            raise MultivariatePolynomialError('polynomial decomposition')

    def shift(f, a):
        if f.ring.is_univariate:
            return f.ring.dup_shift(f, a)
        else:
            raise MultivariatePolynomialError('polynomial shift')

    def sturm(f):
        if f.ring.is_univariate:
            return f.ring.dup_sturm(f)
        else:
            raise MultivariatePolynomialError('sturm sequence')

    def gff_list(f):
        return f.ring.dmp_gff_list(f)

    def sqf_norm(f):
        return f.ring.dmp_sqf_norm(f)

    def sqf_part(f):
        return f.ring.dmp_sqf_part(f)

    def sqf_list(f, all=False):
        return f.ring.dmp_sqf_list(f, all=all)

    def factor_list(f):
        return f.ring.dmp_factor_list(f)