import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
class TestMatrixReturn:

    def test_instance_methods(self):
        a = matrix([1.0], dtype='f8')
        methodargs = {'astype': ('intc',), 'clip': (0.0, 1.0), 'compress': ([1],), 'repeat': (1,), 'reshape': (1,), 'swapaxes': (0, 0), 'dot': np.array([1.0])}
        excluded_methods = ['argmin', 'choose', 'dump', 'dumps', 'fill', 'getfield', 'getA', 'getA1', 'item', 'nonzero', 'put', 'putmask', 'resize', 'searchsorted', 'setflags', 'setfield', 'sort', 'partition', 'argpartition', 'take', 'tofile', 'tolist', 'tostring', 'tobytes', 'all', 'any', 'sum', 'argmax', 'argmin', 'min', 'max', 'mean', 'var', 'ptp', 'prod', 'std', 'ctypes', 'itemset']
        for attrib in dir(a):
            if attrib.startswith('_') or attrib in excluded_methods:
                continue
            f = getattr(a, attrib)
            if isinstance(f, collections.abc.Callable):
                a.astype('f8')
                a.fill(1.0)
                if attrib in methodargs:
                    args = methodargs[attrib]
                else:
                    args = ()
                b = f(*args)
                assert_(type(b) is matrix, '%s' % attrib)
        assert_(type(a.real) is matrix)
        assert_(type(a.imag) is matrix)
        c, d = matrix([0.0]).nonzero()
        assert_(type(c) is np.ndarray)
        assert_(type(d) is np.ndarray)