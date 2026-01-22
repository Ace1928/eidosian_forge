from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterable
from functools import reduce
import re
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .cache import cacheit
from .sorting import default_sort_key
from .kind import NumberKind
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int, func_name, filldedent
from sympy.utilities.iterables import has_variety, sift
from mpmath.libmp import mpf_log, prec_to_dps
from mpmath.libmp.libintmath import giant_steps
from collections import defaultdict
from .mul import Mul
from .add import Add
from .power import Pow
from .function import Function, _derivative_dispatch
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Float, Integer, Rational, _illegal
class ExprBuilder:

    def __init__(self, op, args=None, validator=None, check=True):
        if not hasattr(op, '__call__'):
            raise TypeError('op {} needs to be callable'.format(op))
        self.op = op
        if args is None:
            self.args = []
        else:
            self.args = args
        self.validator = validator
        if validator is not None and check:
            self.validate()

    @staticmethod
    def _build_args(args):
        return [i.build() if isinstance(i, ExprBuilder) else i for i in args]

    def validate(self):
        if self.validator is None:
            return
        args = self._build_args(self.args)
        self.validator(*args)

    def build(self, check=True):
        args = self._build_args(self.args)
        if self.validator and check:
            self.validator(*args)
        return self.op(*args)

    def append_argument(self, arg, check=True):
        self.args.append(arg)
        if self.validator and check:
            self.validate(*self.args)

    def __getitem__(self, item):
        if item == 0:
            return self.op
        else:
            return self.args[item - 1]

    def __repr__(self):
        return str(self.build())

    def search_element(self, elem):
        for i, arg in enumerate(self.args):
            if isinstance(arg, ExprBuilder):
                ret = arg.search_index(elem)
                if ret is not None:
                    return (i,) + ret
            elif id(arg) == id(elem):
                return (i,)
        return None