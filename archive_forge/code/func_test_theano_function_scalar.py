import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_theano_function_scalar():
    """ Test the "scalar" argument to theano_function(). """
    args = [([x, y], [x + y], None, [0]), ([X, Y], [X + Y], None, [2]), ([x, y], [x + y], {x: 0, y: 1}, [1]), ([x, y], [x + y, x - y], None, [0, 0]), ([x, y, X, Y], [x + y, X + Y], None, [0, 2])]
    for inputs, outputs, in_dims, out_dims in args:
        for scalar in [False, True]:
            f = theano_function_(inputs, outputs, dims=in_dims, scalar=scalar)
            assert isinstance(f.theano_function, theano.compile.function_module.Function)
            in_values = [np.ones([1 if bc else 5 for bc in i.type.broadcastable]) for i in f.theano_function.input_storage]
            out_values = f(*in_values)
            if not isinstance(out_values, list):
                out_values = [out_values]
            assert len(out_dims) == len(out_values)
            for d, value in zip(out_dims, out_values):
                if scalar and d == 0:
                    assert isinstance(value, np.number)
                else:
                    assert isinstance(value, np.ndarray)
                    assert value.ndim == d