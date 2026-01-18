import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_platform_dependent_aliases(self):
    if np.int64 is np.int_:
        assert_('int64' in np.int_.__doc__)
    elif np.int64 is np.longlong:
        assert_('int64' in np.longlong.__doc__)