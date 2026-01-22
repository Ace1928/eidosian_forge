import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
def _resolve_op(self, op, dataset, data, flat, expanded, ranges, all_values, keep_index, compute, strict):
    args = op['args']
    fn = op['fn']
    kwargs = dict(op['kwargs'])
    fn_name = self._numpy_funcs.get(fn)
    if fn_name and hasattr(data, fn_name):
        if 'axis' not in kwargs and (not isinstance(fn, np.ufunc)):
            kwargs['axis'] = None
        fn = fn_name
    if isinstance(fn, str):
        accessor = kwargs.pop('accessor', None)
        fn_args = []
    else:
        accessor = False
        fn_args = [data]
    for arg in args:
        if isinstance(arg, dim):
            arg = arg.apply(dataset, flat, expanded, ranges, all_values, keep_index, compute, strict)
        arg = resolve_dependent_value(arg)
        fn_args.append(arg)
    fn_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, dim):
            v = v.apply(dataset, flat, expanded, ranges, all_values, keep_index, compute, strict)
        fn_kwargs[k] = resolve_dependent_value(v)
    args = tuple(fn_args[::-1] if op['reverse'] else fn_args)
    kwargs = dict(fn_kwargs)
    return (fn, fn_name, args, kwargs, accessor)