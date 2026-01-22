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
@classmethod
def _to_nnf(cls, *args, **kwargs):
    simplify = kwargs.get('simplify', True)
    argset = set()
    for arg in args:
        if not is_literal(arg):
            arg = arg.to_nnf(simplify)
        if simplify:
            if isinstance(arg, cls):
                arg = arg.args
            else:
                arg = (arg,)
            for a in arg:
                if Not(a) in argset:
                    return cls.zero
                argset.add(a)
        else:
            argset.add(arg)
    return cls(*argset)