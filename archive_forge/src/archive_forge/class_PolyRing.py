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
class PolyRing(DefaultPrinting, IPolys):
    """Multivariate distributed polynomial ring. """

    def __new__(cls, symbols, domain, order=lex):
        symbols = tuple(_parse_symbols(symbols))
        ngens = len(symbols)
        domain = DomainOpt.preprocess(domain)
        order = OrderOpt.preprocess(order)
        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)
        obj = _ring_cache.get(_hash_tuple)
        if obj is None:
            if domain.is_Composite and set(symbols) & set(domain.symbols):
                raise GeneratorsError("polynomial ring and it's ground domain share generators")
            obj = object.__new__(cls)
            obj._hash_tuple = _hash_tuple
            obj._hash = hash(_hash_tuple)
            obj.dtype = type('PolyElement', (PolyElement,), {'ring': obj})
            obj.symbols = symbols
            obj.ngens = ngens
            obj.domain = domain
            obj.order = order
            obj.zero_monom = (0,) * ngens
            obj.gens = obj._gens()
            obj._gens_set = set(obj.gens)
            obj._one = [(obj.zero_monom, domain.one)]
            if ngens:
                codegen = MonomialOps(ngens)
                obj.monomial_mul = codegen.mul()
                obj.monomial_pow = codegen.pow()
                obj.monomial_mulpow = codegen.mulpow()
                obj.monomial_ldiv = codegen.ldiv()
                obj.monomial_div = codegen.div()
                obj.monomial_lcm = codegen.lcm()
                obj.monomial_gcd = codegen.gcd()
            else:
                monunit = lambda a, b: ()
                obj.monomial_mul = monunit
                obj.monomial_pow = monunit
                obj.monomial_mulpow = lambda a, b, c: ()
                obj.monomial_ldiv = monunit
                obj.monomial_div = monunit
                obj.monomial_lcm = monunit
                obj.monomial_gcd = monunit
            if order is lex:
                obj.leading_expv = max
            else:
                obj.leading_expv = lambda f: max(f, key=order)
            for symbol, generator in zip(obj.symbols, obj.gens):
                if isinstance(symbol, Symbol):
                    name = symbol.name
                    if not hasattr(obj, name):
                        setattr(obj, name, generator)
            _ring_cache[_hash_tuple] = obj
        return obj

    def _gens(self):
        """Return a list of polynomial generators. """
        one = self.domain.one
        _gens = []
        for i in range(self.ngens):
            expv = self.monomial_basis(i)
            poly = self.zero
            poly[expv] = one
            _gens.append(poly)
        return tuple(_gens)

    def __getnewargs__(self):
        return (self.symbols, self.domain, self.order)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['leading_expv']
        for key, value in state.items():
            if key.startswith('monomial_'):
                del state[key]
        return state

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, PolyRing) and (self.symbols, self.domain, self.ngens, self.order) == (other.symbols, other.domain, other.ngens, other.order)

    def __ne__(self, other):
        return not self == other

    def clone(self, symbols=None, domain=None, order=None):
        return self.__class__(symbols or self.symbols, domain or self.domain, order or self.order)

    def monomial_basis(self, i):
        """Return the ith-basis element. """
        basis = [0] * self.ngens
        basis[i] = 1
        return tuple(basis)

    @property
    def zero(self):
        return self.dtype()

    @property
    def one(self):
        return self.dtype(self._one)

    def domain_new(self, element, orig_domain=None):
        return self.domain.convert(element, orig_domain)

    def ground_new(self, coeff):
        return self.term_new(self.zero_monom, coeff)

    def term_new(self, monom, coeff):
        coeff = self.domain_new(coeff)
        poly = self.zero
        if coeff:
            poly[monom] = coeff
        return poly

    def ring_new(self, element):
        if isinstance(element, PolyElement):
            if self == element.ring:
                return element
            elif isinstance(self.domain, PolynomialRing) and self.domain.ring == element.ring:
                return self.ground_new(element)
            else:
                raise NotImplementedError('conversion')
        elif isinstance(element, str):
            raise NotImplementedError('parsing')
        elif isinstance(element, dict):
            return self.from_dict(element)
        elif isinstance(element, list):
            try:
                return self.from_terms(element)
            except ValueError:
                return self.from_list(element)
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)
    __call__ = ring_new

    def from_dict(self, element, orig_domain=None):
        domain_new = self.domain_new
        poly = self.zero
        for monom, coeff in element.items():
            coeff = domain_new(coeff, orig_domain)
            if coeff:
                poly[monom] = coeff
        return poly

    def from_terms(self, element, orig_domain=None):
        return self.from_dict(dict(element), orig_domain)

    def from_list(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens - 1, self.domain))

    def _rebuild_expr(self, expr, mapping):
        domain = self.domain

        def _rebuild(expr):
            generator = mapping.get(expr)
            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            else:
                base, exp = expr.as_base_exp()
                if exp.is_Integer and exp > 1:
                    return _rebuild(base) ** int(exp)
                else:
                    return self.ground_new(domain.convert(expr))
        return _rebuild(sympify(expr))

    def from_expr(self, expr):
        mapping = dict(list(zip(self.symbols, self.gens)))
        try:
            poly = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            raise ValueError('expected an expression convertible to a polynomial in %s, got %s' % (self, expr))
        else:
            return self.ring_new(poly)

    def index(self, gen):
        """Compute index of ``gen`` in ``self.gens``. """
        if gen is None:
            if self.ngens:
                i = 0
            else:
                i = -1
        elif isinstance(gen, int):
            i = gen
            if 0 <= i and i < self.ngens:
                pass
            elif -self.ngens <= i and i <= -1:
                i = -i - 1
            else:
                raise ValueError('invalid generator index: %s' % gen)
        elif isinstance(gen, self.dtype):
            try:
                i = self.gens.index(gen)
            except ValueError:
                raise ValueError('invalid generator: %s' % gen)
        elif isinstance(gen, str):
            try:
                i = self.symbols.index(gen)
            except ValueError:
                raise ValueError('invalid generator: %s' % gen)
        else:
            raise ValueError('expected a polynomial generator, an integer, a string or None, got %s' % gen)
        return i

    def drop(self, *gens):
        """Remove specified generators from this ring. """
        indices = set(map(self.index, gens))
        symbols = [s for i, s in enumerate(self.symbols) if i not in indices]
        if not symbols:
            return self.domain
        else:
            return self.clone(symbols=symbols)

    def __getitem__(self, key):
        symbols = self.symbols[key]
        if not symbols:
            return self.domain
        else:
            return self.clone(symbols=symbols)

    def to_ground(self):
        if self.domain.is_Composite or hasattr(self.domain, 'domain'):
            return self.clone(domain=self.domain.domain)
        else:
            raise ValueError('%s is not a composite domain' % self.domain)

    def to_domain(self):
        return PolynomialRing(self)

    def to_field(self):
        from sympy.polys.fields import FracField
        return FracField(self.symbols, self.domain, self.order)

    @property
    def is_univariate(self):
        return len(self.gens) == 1

    @property
    def is_multivariate(self):
        return len(self.gens) > 1

    def add(self, *objs):
        """
        Add a sequence of polynomials or containers of polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> R, x = ring("x", ZZ)
        >>> R.add([ x**2 + 2*i + 3 for i in range(4) ])
        4*x**2 + 24
        >>> _.factor_list()
        (4, [(x**2 + 6, 1)])

        """
        p = self.zero
        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                p += self.add(*obj)
            else:
                p += obj
        return p

    def mul(self, *objs):
        """
        Multiply a sequence of polynomials or containers of polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> R, x = ring("x", ZZ)
        >>> R.mul([ x**2 + 2*i + 3 for i in range(4) ])
        x**8 + 24*x**6 + 206*x**4 + 744*x**2 + 945
        >>> _.factor_list()
        (1, [(x**2 + 3, 1), (x**2 + 5, 1), (x**2 + 7, 1), (x**2 + 9, 1)])

        """
        p = self.one
        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                p *= self.mul(*obj)
            else:
                p *= obj
        return p

    def drop_to_ground(self, *gens):
        """
        Remove specified generators from the ring and inject them into
        its domain.
        """
        indices = set(map(self.index, gens))
        symbols = [s for i, s in enumerate(self.symbols) if i not in indices]
        gens = [gen for i, gen in enumerate(self.gens) if i not in indices]
        if not symbols:
            return self
        else:
            return self.clone(symbols=symbols, domain=self.drop(*gens))

    def compose(self, other):
        """Add the generators of ``other`` to ``self``"""
        if self != other:
            syms = set(self.symbols).union(set(other.symbols))
            return self.clone(symbols=list(syms))
        else:
            return self

    def add_gens(self, symbols):
        """Add the elements of ``symbols`` as generators to ``self``"""
        syms = set(self.symbols).union(set(symbols))
        return self.clone(symbols=list(syms))

    def symmetric_poly(self, n):
        """
        Return the elementary symmetric polynomial of degree *n* over
        this ring's generators.
        """
        if n < 0 or n > self.ngens:
            raise ValueError('Cannot generate symmetric polynomial of order %s for %s' % (n, self.gens))
        elif not n:
            return self.one
        else:
            poly = self.zero
            for s in subsets(range(self.ngens), int(n)):
                monom = tuple((int(i in s) for i in range(self.ngens)))
                poly += self.term_new(monom, self.domain.one)
            return poly