import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_recarrays(self):
    """Test record arrays."""
    a = np.empty(2, [('floupi', float), ('floupa', float)])
    a['floupi'] = [1, 2]
    a['floupa'] = [1, 2]
    b = a.copy()
    self._test_equal(a, b)
    c = np.empty(2, [('floupipi', float), ('floupi', float), ('floupa', float)])
    c['floupipi'] = a['floupi'].copy()
    c['floupa'] = a['floupa'].copy()
    with pytest.raises(TypeError):
        self._test_not_equal(c, b)