import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
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