from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from itertools import product
import operator
from .sympify import sympify
from .basic import Basic
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
def _eval_real_imag(self, real):
    zero = False
    t_not_re_im = None
    for t in self.args:
        if (t.is_complex or t.is_infinite) is False and t.is_extended_real is False:
            return False
        elif t.is_imaginary:
            real = not real
        elif t.is_extended_real:
            if not zero:
                z = t.is_zero
                if not z and zero is False:
                    zero = z
                elif z:
                    if all((a.is_finite for a in self.args)):
                        return True
                    return
        elif t.is_extended_real is False:
            if t_not_re_im:
                return
            t_not_re_im = t
        elif t.is_imaginary is False:
            if t_not_re_im:
                return
            t_not_re_im = t
        else:
            return
    if t_not_re_im:
        if t_not_re_im.is_extended_real is False:
            if real:
                return zero
        if t_not_re_im.is_imaginary is False:
            if not real:
                return zero
    elif zero is False:
        return real
    elif real:
        return real