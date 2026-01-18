from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
def wicks(e, **kw_args):
    """
    Returns the normal ordered equivalent of an expression using Wicks Theorem.

    Examples
    ========

    >>> from sympy import symbols, Dummy
    >>> from sympy.physics.secondquant import wicks, F, Fd
    >>> p, q, r = symbols('p,q,r')
    >>> wicks(Fd(p)*F(q))
    KroneckerDelta(_i, q)*KroneckerDelta(p, q) + NO(CreateFermion(p)*AnnihilateFermion(q))

    By default, the expression is expanded:

    >>> wicks(F(p)*(F(q)+F(r)))
    NO(AnnihilateFermion(p)*AnnihilateFermion(q)) + NO(AnnihilateFermion(p)*AnnihilateFermion(r))

    With the keyword 'keep_only_fully_contracted=True', only fully contracted
    terms are returned.

    By request, the result can be simplified in the following order:
     -- KroneckerDelta functions are evaluated
     -- Dummy variables are substituted consistently across terms

    >>> p, q, r = symbols('p q r', cls=Dummy)
    >>> wicks(Fd(p)*(F(q)+F(r)), keep_only_fully_contracted=True)
    KroneckerDelta(_i, _q)*KroneckerDelta(_p, _q) + KroneckerDelta(_i, _r)*KroneckerDelta(_p, _r)

    """
    if not e:
        return S.Zero
    opts = {'simplify_kronecker_deltas': False, 'expand': True, 'simplify_dummies': False, 'keep_only_fully_contracted': False}
    opts.update(kw_args)
    if isinstance(e, NO):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e
    elif isinstance(e, FermionicOperator):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e
    e = e.doit(wicks=True)
    e = e.expand()
    if isinstance(e, Add):
        if opts['simplify_dummies']:
            return substitute_dummies(Add(*[wicks(term, **kw_args) for term in e.args]))
        else:
            return Add(*[wicks(term, **kw_args) for term in e.args])
    if isinstance(e, Mul):
        c_part = []
        string1 = []
        for factor in e.args:
            if factor.is_commutative:
                c_part.append(factor)
            else:
                string1.append(factor)
        n = len(string1)
        if n == 0:
            result = e
        elif n == 1:
            if opts['keep_only_fully_contracted']:
                return S.Zero
            else:
                result = e
        else:
            if isinstance(string1[0], BosonicOperator):
                raise NotImplementedError
            string1 = tuple(string1)
            result = _get_contractions(string1, keep_only_fully_contracted=opts['keep_only_fully_contracted'])
            result = Mul(*c_part) * result
        if opts['expand']:
            result = result.expand()
        if opts['simplify_kronecker_deltas']:
            result = evaluate_deltas(result)
        return result
    return e