import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_theano_function_kwargs():
    """
    Test passing additional kwargs from theano_function() to theano.function().
    """
    import numpy as np
    f = theano_function_([x, y, z], [x + y], dim=1, on_unused_input='ignore', dtypes={x: 'float64', y: 'float64', z: 'float64'})
    assert np.linalg.norm(f([1, 2], [3, 4], [0, 0]) - np.asarray([4, 6])) < 1e-09
    f = theano_function_([x, y, z], [x + y], dtypes={x: 'float64', y: 'float64', z: 'float64'}, dim=1, on_unused_input='ignore')
    xx = np.arange(3).astype('float64')
    yy = 2 * np.arange(3).astype('float64')
    zz = 2 * np.arange(3).astype('float64')
    assert np.linalg.norm(f(xx, yy, zz) - 3 * np.arange(3)) < 1e-09