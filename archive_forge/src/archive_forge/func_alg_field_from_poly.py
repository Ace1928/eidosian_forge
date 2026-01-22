from __future__ import annotations
from typing import Any
from sympy.core.numbers import AlgebraicNumber
from sympy.core import Basic, sympify
from sympy.core.sorting import default_sort_key, ordered
from sympy.external.gmpy import HAS_GMPY
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import UnificationFailed, CoercionFailed, DomainError
from sympy.polys.polyutils import _unify_gens, _not_a_coeff
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
def alg_field_from_poly(self, poly, alias=None, root_index=-1):
    """
        Convenience method to construct an algebraic extension on a root of a
        polynomial, chosen by root index.

        Parameters
        ==========

        poly : :py:class:`~.Poly`
            The polynomial whose root generates the extension.
        alias : str, optional (default=None)
            Symbol name for the generator of the extension.
            E.g. "alpha" or "theta".
        root_index : int, optional (default=-1)
            Specifies which root of the polynomial is desired. The ordering is
            as defined by the :py:class:`~.ComplexRootOf` class. The default of
            ``-1`` selects the most natural choice in the common cases of
            quadratic and cyclotomic fields (the square root on the positive
            real or imaginary axis, resp. $\\mathrm{e}^{2\\pi i/n}$).

        Examples
        ========

        >>> from sympy import QQ, Poly
        >>> from sympy.abc import x
        >>> f = Poly(x**2 - 2)
        >>> K = QQ.alg_field_from_poly(f)
        >>> K.ext.minpoly == f
        True
        >>> g = Poly(8*x**3 - 6*x - 1)
        >>> L = QQ.alg_field_from_poly(g, "alpha")
        >>> L.ext.minpoly == g
        True
        >>> L.to_sympy(L([1, 1, 1]))
        alpha**2 + alpha + 1

        """
    from sympy.polys.rootoftools import CRootOf
    root = CRootOf(poly, root_index)
    alpha = AlgebraicNumber(root, alias=alias)
    return self.algebraic_field(alpha, alias=alias)