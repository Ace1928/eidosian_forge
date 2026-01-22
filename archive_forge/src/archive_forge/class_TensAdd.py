from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
class TensAdd(TensExpr, AssocOp):
    """
    Sum of tensors.

    Parameters
    ==========

    free_args : list of the free indices

    Attributes
    ==========

    ``args`` : tuple of addends
    ``rank`` : rank of the tensor
    ``free_args`` : list of the free indices in sorted order

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_heads, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> a, b = tensor_indices('a,b', Lorentz)
    >>> p, q = tensor_heads('p,q', [Lorentz])
    >>> t = p(a) + q(a); t
    p(a) + q(a)

    Examples with components data added to the tensor expression:

    >>> from sympy import symbols, diag
    >>> x, y, z, t = symbols("x y z t")
    >>> repl = {}
    >>> repl[Lorentz] = diag(1, -1, -1, -1)
    >>> repl[p(a)] = [1, 2, 3, 4]
    >>> repl[q(a)] = [x, y, z, t]

    The following are: 2**2 - 3**2 - 2**2 - 7**2 ==> -58

    >>> expr = p(a) + q(a)
    >>> expr.replace_with_arrays(repl, [a])
    [x + 1, y + 2, z + 3, t + 4]
    """

    def __new__(cls, *args, **kw_args):
        args = [_sympify(x) for x in args if x]
        args = TensAdd._tensAdd_flatten(args)
        args.sort(key=default_sort_key)
        if not args:
            return S.Zero
        if len(args) == 1:
            return args[0]
        return Basic.__new__(cls, *args, **kw_args)

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    def get_free_indices(self) -> list[TensorIndex]:
        return self.free_indices

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        newargs = [arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg in self.args]
        return self.func(*newargs)

    @memoize_property
    def rank(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].rank
        else:
            return 0

    @memoize_property
    def free_args(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].free_args
        else:
            return []

    @memoize_property
    def free_indices(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].get_free_indices()
        else:
            return set()

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        args = [arg for arg in args if arg != S.Zero]
        if len(args) == 0:
            return S.Zero
        elif len(args) == 1:
            return args[0]
        TensAdd._tensAdd_check(args)
        args = TensAdd._tensAdd_collect_terms(args)

        def sort_key(t):
            if not isinstance(t, TensExpr):
                return ([], [], [])
            if hasattr(t, '_index_structure') and hasattr(t, 'components'):
                x = get_index_structure(t)
                return (t.components, x.free, x.dum)
            return ([], [], [])
        args.sort(key=sort_key)
        if not args:
            return S.Zero
        if len(args) == 1:
            return args[0]
        obj = self.func(*args)
        return obj

    @staticmethod
    def _tensAdd_flatten(args):
        a = []
        for x in args:
            if isinstance(x, (Add, TensAdd)):
                a.extend(list(x.args))
            else:
                a.append(x)
        args = [x for x in a if x.coeff]
        return args

    @staticmethod
    def _tensAdd_check(args):

        def get_indices_set(x: Expr) -> set[TensorIndex]:
            if isinstance(x, TensExpr):
                return set(x.get_free_indices())
            return set()
        indices0 = get_indices_set(args[0])
        list_indices = [get_indices_set(arg) for arg in args[1:]]
        if not all((x == indices0 for x in list_indices)):
            raise ValueError('all tensors must have the same indices')

    @staticmethod
    def _tensAdd_collect_terms(args):
        terms_dict = defaultdict(list)
        scalars = S.Zero
        if isinstance(args[0], TensExpr):
            free_indices = set(args[0].get_free_indices())
        else:
            free_indices = set()
        for arg in args:
            if not isinstance(arg, TensExpr):
                if free_indices != set():
                    raise ValueError('wrong valence')
                scalars += arg
                continue
            if free_indices != set(arg.get_free_indices()):
                raise ValueError('wrong valence')
            terms_dict[arg.nocoeff].append(arg.coeff)
        new_args = [TensMul(Add(*coeff), t).doit() for t, coeff in terms_dict.items() if Add(*coeff) != 0]
        if isinstance(scalars, Add):
            new_args = list(scalars.args) + new_args
        elif scalars != 0:
            new_args = [scalars] + new_args
        return new_args

    def get_indices(self):
        indices = []
        for arg in self.args:
            indices.extend([i for i in get_indices(arg) if i not in indices])
        return indices

    def _expand(self, **hints):
        return TensAdd(*[_expand(i, **hints) for i in self.args])

    def __call__(self, *indices):
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        if indices == free_args:
            return self
        index_tuples = list(zip(free_args, indices))
        a = [x.func(*x.substitute_indices(*index_tuples).args) for x in self.args]
        res = TensAdd(*a).doit()
        return res

    def canon_bp(self):
        """
        Canonicalize using the Butler-Portugal algorithm for canonicalization
        under monoterm symmetries.
        """
        expr = self.expand()
        args = [canon_bp(x) for x in expr.args]
        res = TensAdd(*args).doit()
        return res

    def equals(self, other):
        other = _sympify(other)
        if isinstance(other, TensMul) and other.coeff == 0:
            return all((x.coeff == 0 for x in self.args))
        if isinstance(other, TensExpr):
            if self.rank != other.rank:
                return False
        if isinstance(other, TensAdd):
            if set(self.args) != set(other.args):
                return False
            else:
                return True
        t = self - other
        if not isinstance(t, TensExpr):
            return t == 0
        elif isinstance(t, TensMul):
            return t.coeff == 0
        else:
            return all((x.coeff == 0 for x in t.args))

    def __getitem__(self, item):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    def contract_delta(self, delta):
        args = [x.contract_delta(delta) for x in self.args]
        t = TensAdd(*args).doit()
        return canon_bp(t)

    def contract_metric(self, g):
        """
        Raise or lower indices with the metric ``g``.

        Parameters
        ==========

        g :  metric

        contract_all : if True, eliminate all ``g`` which are contracted

        Notes
        =====

        see the ``TensorIndexType`` docstring for the contraction conventions
        """
        args = [contract_metric(x, g) for x in self.args]
        t = TensAdd(*args).doit()
        return canon_bp(t)

    def substitute_indices(self, *index_tuples):
        new_args = []
        for arg in self.args:
            if isinstance(arg, TensExpr):
                arg = arg.substitute_indices(*index_tuples)
            new_args.append(arg)
        return TensAdd(*new_args).doit()

    def _print(self):
        a = []
        args = self.args
        for x in args:
            a.append(str(x))
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _extract_data(self, replacement_dict):
        from sympy.tensor.array import Array, permutedims
        args_indices, arrays = zip(*[arg._extract_data(replacement_dict) if isinstance(arg, TensExpr) else ([], arg) for arg in self.args])
        arrays = [Array(i) for i in arrays]
        ref_indices = args_indices[0]
        for i in range(1, len(args_indices)):
            indices = args_indices[i]
            array = arrays[i]
            permutation = TensMul._get_indices_permutation(indices, ref_indices)
            arrays[i] = permutedims(array, permutation)
        return (ref_indices, sum(arrays, Array.zeros(*array.shape)))

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self.expand()]

    @data.setter
    def data(self, data):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            _tensor_data_substitution_dict[self] = data

    @data.deleter
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]

    def __iter__(self):
        deprecate_data()
        if not self.data:
            raise ValueError('No iteration on abstract tensors')
        return self.data.flatten().__iter__()

    def _eval_rewrite_as_Indexed(self, *args):
        return Add.fromiter(args)

    def _eval_partial_derivative(self, s):
        list_addends = []
        for a in self.args:
            if isinstance(a, TensExpr):
                list_addends.append(a._eval_partial_derivative(s))
            elif s._diff_wrt:
                list_addends.append(a._eval_derivative(s))
        return self.func(*list_addends)