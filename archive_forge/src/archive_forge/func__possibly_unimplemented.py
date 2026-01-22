import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def _possibly_unimplemented(cls, require=True):
    """
    Construct a class that either runs tests as usual (require=True),
    or each method skips if it encounters a common error.
    """
    if require:
        return cls
    else:

        def wrap(fc):

            @functools.wraps(fc)
            def wrapper(*a, **kw):
                try:
                    return fc(*a, **kw)
                except (NotImplementedError, TypeError, ValueError, IndexError, AttributeError):
                    raise pytest.skip('feature not implemented')
            return wrapper
        new_dict = dict(cls.__dict__)
        for name, func in cls.__dict__.items():
            if name.startswith('test_'):
                new_dict[name] = wrap(func)
        return type(cls.__name__ + 'NotImplemented', cls.__bases__, new_dict)