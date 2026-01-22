from __future__ import annotations
from .basic import Atom, Basic
from .sorting import ordered
from .evalf import EvalfMixin
from .function import AppliedUndef
from .singleton import S
from .sympify import _sympify, SympifyError
from .parameters import global_parameters
from .logic import fuzzy_bool, fuzzy_xor, fuzzy_and, fuzzy_not
from sympy.logic.boolalg import Boolean, BooleanAtom
from sympy.utilities.iterables import sift
from sympy.utilities.misc import filldedent
from .expr import Expr
from sympy.multipledispatch import dispatch
from .containers import Tuple
from .symbol import Symbol
def _eval_rewrite_as_Add(self, *args, **kwargs):
    """
        return Eq(L, R) as L - R. To control the evaluation of
        the result set pass `evaluate=True` to give L - R;
        if `evaluate=None` then terms in L and R will not cancel
        but they will be listed in canonical order; otherwise
        non-canonical args will be returned. If one side is 0, the
        non-zero side will be returned.

        Examples
        ========

        >>> from sympy import Eq, Add
        >>> from sympy.abc import b, x
        >>> eq = Eq(x + b, x - b)
        >>> eq.rewrite(Add)
        2*b
        >>> eq.rewrite(Add, evaluate=None).args
        (b, b, x, -x)
        >>> eq.rewrite(Add, evaluate=False).args
        (b, x, b, -x)
        """
    from .add import _unevaluated_Add, Add
    L, R = args
    if L == 0:
        return R
    if R == 0:
        return L
    evaluate = kwargs.get('evaluate', True)
    if evaluate:
        return L - R
    args = Add.make_args(L) + Add.make_args(-R)
    if evaluate is None:
        return _unevaluated_Add(*args)
    return Add._from_args(args)