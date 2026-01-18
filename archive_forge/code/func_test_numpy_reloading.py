from numpy.testing import (
from numpy.compat import pickle
import pytest
import sys
import subprocess
import textwrap
from importlib import reload
def test_numpy_reloading():
    import numpy as np
    import numpy._globals
    _NoValue = np._NoValue
    VisibleDeprecationWarning = np.VisibleDeprecationWarning
    ModuleDeprecationWarning = np.ModuleDeprecationWarning
    with assert_warns(UserWarning):
        reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is np.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is np.VisibleDeprecationWarning)
    assert_raises(RuntimeError, reload, numpy._globals)
    with assert_warns(UserWarning):
        reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is np.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is np.VisibleDeprecationWarning)