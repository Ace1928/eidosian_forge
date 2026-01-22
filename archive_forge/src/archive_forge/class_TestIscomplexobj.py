import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestIscomplexobj:

    def test_basic(self):
        z = np.array([-1, 0, 1])
        assert_(not iscomplexobj(z))
        z = np.array([-1j, 0, -1])
        assert_(iscomplexobj(z))

    def test_scalar(self):
        assert_(not iscomplexobj(1.0))
        assert_(iscomplexobj(1 + 0j))

    def test_list(self):
        assert_(iscomplexobj([3, 1 + 0j, True]))
        assert_(not iscomplexobj([3, 1, True]))

    def test_duck(self):

        class DummyComplexArray:

            @property
            def dtype(self):
                return np.dtype(complex)
        dummy = DummyComplexArray()
        assert_(iscomplexobj(dummy))

    def test_pandas_duck(self):

        class PdComplex(np.complex128):
            pass

        class PdDtype:
            name = 'category'
            names = None
            type = PdComplex
            kind = 'c'
            str = '<c16'
            base = np.dtype('complex128')

        class DummyPd:

            @property
            def dtype(self):
                return PdDtype
        dummy = DummyPd()
        assert_(iscomplexobj(dummy))

    def test_custom_dtype_duck(self):

        class MyArray(list):

            @property
            def dtype(self):
                return complex
        a = MyArray([1 + 0j, 2 + 0j, 3 + 0j])
        assert_(iscomplexobj(a))