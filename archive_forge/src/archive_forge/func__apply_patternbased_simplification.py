from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def _apply_patternbased_simplification(rv, patterns, measure, dominatingvalue, replacementvalue=None, threeterm_patterns=None):
    """
    Replace patterns of Relational

    Parameters
    ==========

    rv : Expr
        Boolean expression

    patterns : tuple
        Tuple of tuples, with (pattern to simplify, simplified pattern) with
        two terms.

    measure : function
        Simplification measure.

    dominatingvalue : Boolean or ``None``
        The dominating value for the function of consideration.
        For example, for :py:class:`~.And` ``S.false`` is dominating.
        As soon as one expression is ``S.false`` in :py:class:`~.And`,
        the whole expression is ``S.false``.

    replacementvalue : Boolean or ``None``, optional
        The resulting value for the whole expression if one argument
        evaluates to ``dominatingvalue``.
        For example, for :py:class:`~.Nand` ``S.false`` is dominating, but
        in this case the resulting value is ``S.true``. Default is ``None``.
        If ``replacementvalue`` is ``None`` and ``dominatingvalue`` is not
        ``None``, ``replacementvalue = dominatingvalue``.

    threeterm_patterns : tuple, optional
        Tuple of tuples, with (pattern to simplify, simplified pattern) with
        three terms.

    """
    from sympy.core.relational import Relational, _canonical
    if replacementvalue is None and dominatingvalue is not None:
        replacementvalue = dominatingvalue
    Rel, nonRel = sift(rv.args, lambda i: isinstance(i, Relational), binary=True)
    if len(Rel) <= 1:
        return rv
    Rel, nonRealRel = sift(Rel, lambda i: not any((s.is_real is False for s in i.free_symbols)), binary=True)
    Rel = [i.canonical for i in Rel]
    if threeterm_patterns and len(Rel) >= 3:
        Rel = _apply_patternbased_threeterm_simplification(Rel, threeterm_patterns, rv.func, dominatingvalue, replacementvalue, measure)
    Rel = _apply_patternbased_twoterm_simplification(Rel, patterns, rv.func, dominatingvalue, replacementvalue, measure)
    rv = rv.func(*[_canonical(i) for i in ordered(Rel)] + nonRel + nonRealRel)
    return rv