import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
@pytest.mark.skipif(sys.flags.optimize > 1, reason='no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1')
@pytest.mark.xfail(IS_PYPY, reason='PyPy cannot modify tp_doc after PyType_Ready')
class TestDocStrings:

    def test_platform_dependent_aliases(self):
        if np.int64 is np.int_:
            assert_('int64' in np.int_.__doc__)
        elif np.int64 is np.longlong:
            assert_('int64' in np.longlong.__doc__)