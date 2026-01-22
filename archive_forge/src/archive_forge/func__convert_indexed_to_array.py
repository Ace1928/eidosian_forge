from collections import defaultdict
from sympy import Function
from sympy.combinatorics.permutations import _af_invert
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, \
from sympy.tensor.array.expressions.utils import _get_argindex, _get_diagonal_indices
def _convert_indexed_to_array(expr):
    if isinstance(expr, Sum):
        function = expr.function
        summation_indices = expr.variables
        subexpr, subindices = _convert_indexed_to_array(function)
        subindicessets = {j: i for i in subindices if isinstance(i, frozenset) for j in i}
        summation_indices = sorted({subindicessets.get(i, i) for i in summation_indices}, key=default_sort_key)
        kronecker_indices = set()
        if isinstance(function, Mul):
            for arg in function.args:
                if not isinstance(arg, KroneckerDelta):
                    continue
                arg_indices = sorted(set(arg.indices), key=default_sort_key)
                if len(arg_indices) == 2:
                    kronecker_indices.update(arg_indices)
        kronecker_indices = sorted(kronecker_indices, key=default_sort_key)
        shape = get_shape(subexpr)
        if shape:
            for ind, istart, iend in expr.limits:
                i = _get_argindex(subindices, ind)
                if istart != 0 or iend + 1 != shape[i]:
                    raise ValueError('summation index and array dimension mismatch: %s' % ind)
        contraction_indices = []
        subindices = list(subindices)
        if isinstance(subexpr, ArrayDiagonal):
            diagonal_indices = list(subexpr.diagonal_indices)
            dindices = subindices[-len(diagonal_indices):]
            subindices = subindices[:-len(diagonal_indices)]
            for index in summation_indices:
                if index in dindices:
                    position = dindices.index(index)
                    contraction_indices.append(diagonal_indices[position])
                    diagonal_indices[position] = None
            diagonal_indices = [i for i in diagonal_indices if i is not None]
            for i, ind in enumerate(subindices):
                if ind in summation_indices:
                    pass
            if diagonal_indices:
                subexpr = _array_diagonal(subexpr.expr, *diagonal_indices)
            else:
                subexpr = subexpr.expr
        axes_contraction = defaultdict(list)
        for i, ind in enumerate(subindices):
            include = all((j not in kronecker_indices for j in ind)) if isinstance(ind, frozenset) else ind not in kronecker_indices
            if ind in summation_indices and include:
                axes_contraction[ind].append(i)
                subindices[i] = None
        for k, v in axes_contraction.items():
            if any((i in kronecker_indices for i in k)) if isinstance(k, frozenset) else k in kronecker_indices:
                continue
            contraction_indices.append(tuple(v))
        free_indices = [i for i in subindices if i is not None]
        indices_ret = list(free_indices)
        indices_ret.sort(key=lambda x: free_indices.index(x))
        return (_array_contraction(subexpr, *contraction_indices, free_indices=free_indices), tuple(indices_ret))
    if isinstance(expr, Mul):
        args, indices = zip(*[_convert_indexed_to_array(arg) for arg in expr.args])
        kronecker_delta_repl = {}
        for arg in args:
            if not isinstance(arg, KroneckerDelta):
                continue
            i, j = arg.indices
            kindices = set(arg.indices)
            if i in kronecker_delta_repl:
                kindices.update(kronecker_delta_repl[i])
            if j in kronecker_delta_repl:
                kindices.update(kronecker_delta_repl[j])
            kindices = frozenset(kindices)
            for index in kindices:
                kronecker_delta_repl[index] = kindices
        newargs = []
        newindices = []
        for arg, loc_indices in zip(args, indices):
            if isinstance(arg, KroneckerDelta):
                continue
            newargs.append(arg)
            newindices.append(loc_indices)
        flattened_indices = [kronecker_delta_repl.get(j, j) for i in newindices for j in i]
        diagonal_indices, ret_indices = _get_diagonal_indices(flattened_indices)
        tp = _array_tensor_product(*newargs)
        if diagonal_indices:
            return (_array_diagonal(tp, *diagonal_indices), ret_indices)
        else:
            return (tp, ret_indices)
    if isinstance(expr, MatrixElement):
        indices = expr.args[1:]
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return (_array_diagonal(expr.args[0], *diagonal_indices), ret_indices)
        else:
            return (expr.args[0], ret_indices)
    if isinstance(expr, ArrayElement):
        indices = expr.indices
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return (_array_diagonal(expr.name, *diagonal_indices), ret_indices)
        else:
            return (expr.name, ret_indices)
    if isinstance(expr, Indexed):
        indices = expr.indices
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return (_array_diagonal(expr.base, *diagonal_indices), ret_indices)
        else:
            return (expr.args[0], ret_indices)
    if isinstance(expr, IndexedBase):
        raise NotImplementedError
    if isinstance(expr, KroneckerDelta):
        return (expr, expr.indices)
    if isinstance(expr, Add):
        args, indices = zip(*[_convert_indexed_to_array(arg) for arg in expr.args])
        args = list(args)
        index0 = []
        shape0 = []
        for arg, arg_indices in zip(args, indices):
            arg_indices_set = set(arg_indices)
            arg_indices_missing = arg_indices_set.difference(index0)
            index0.extend([i for i in arg_indices if i in arg_indices_missing])
            arg_shape = get_shape(arg)
            shape0.extend([arg_shape[i] for i, e in enumerate(arg_indices) if e in arg_indices_missing])
        for i, (arg, arg_indices) in enumerate(zip(args, indices)):
            if len(arg_indices) < len(index0):
                missing_indices_pos = [i for i, e in enumerate(index0) if e not in arg_indices]
                missing_shape = [shape0[i] for i in missing_indices_pos]
                arg_indices = tuple((index0[j] for j in missing_indices_pos)) + arg_indices
                args[i] = _array_tensor_product(OneArray(*missing_shape), args[i])
            permutation = Permutation([arg_indices.index(j) for j in index0])
            args[i] = _permute_dims(args[i], permutation)
        return (_array_add(*args), tuple(index0))
    if isinstance(expr, Pow):
        subexpr, subindices = _convert_indexed_to_array(expr.base)
        if isinstance(expr.exp, (int, Integer)):
            diags = zip(*[(2 * i, 2 * i + 1) for i in range(expr.exp)])
            arr = _array_diagonal(_array_tensor_product(*[subexpr for i in range(expr.exp)]), *diags)
            return (arr, subindices)
    if isinstance(expr, Function):
        subexpr, subindices = _convert_indexed_to_array(expr.args[0])
        return (ArrayElementwiseApplyFunc(type(expr), subexpr), subindices)
    return (expr, ())