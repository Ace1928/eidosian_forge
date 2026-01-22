from __future__ import annotations
import collections
from functools import reduce
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.core.expr import Expr
from sympy.core.power import Pow
def _get_dimensional_dependencies_for_name(self, dimension):
    if isinstance(dimension, str):
        dimension = Dimension(Symbol(dimension))
    elif not isinstance(dimension, Dimension):
        dimension = Dimension(dimension)
    if dimension.name.is_Symbol:
        return dict(self.dimensional_dependencies.get(dimension, {dimension: 1}))
    if dimension.name.is_number or dimension.name.is_NumberSymbol:
        return {}
    get_for_name = self._get_dimensional_dependencies_for_name
    if dimension.name.is_Mul:
        ret = collections.defaultdict(int)
        dicts = [get_for_name(i) for i in dimension.name.args]
        for d in dicts:
            for k, v in d.items():
                ret[k] += v
        return {k: v for k, v in ret.items() if v != 0}
    if dimension.name.is_Add:
        dicts = [get_for_name(i) for i in dimension.name.args]
        if all((d == dicts[0] for d in dicts[1:])):
            return dicts[0]
        raise TypeError('Only equivalent dimensions can be added or subtracted.')
    if dimension.name.is_Pow:
        dim_base = get_for_name(dimension.name.base)
        dim_exp = get_for_name(dimension.name.exp)
        if dim_exp == {} or dimension.name.exp.is_Symbol:
            return {k: v * dimension.name.exp for k, v in dim_base.items()}
        else:
            raise TypeError('The exponent for the power operator must be a Symbol or dimensionless.')
    if dimension.name.is_Function:
        args = (Dimension._from_dimensional_dependencies(get_for_name(arg)) for arg in dimension.name.args)
        result = dimension.name.func(*args)
        dicts = [get_for_name(i) for i in dimension.name.args]
        if isinstance(result, Dimension):
            return self.get_dimensional_dependencies(result)
        elif result.func == dimension.name.func:
            if isinstance(dimension.name, TrigonometricFunction):
                if dicts[0] in ({}, {Dimension('angle'): 1}):
                    return {}
                else:
                    raise TypeError('The input argument for the function {} must be dimensionless or have dimensions of angle.'.format(dimension.func))
            elif all((item == {} for item in dicts)):
                return {}
            else:
                raise TypeError('The input arguments for the function {} must be dimensionless.'.format(dimension.func))
        else:
            return get_for_name(result)
    raise TypeError('Type {} not implemented for get_dimensional_dependencies'.format(type(dimension.name)))