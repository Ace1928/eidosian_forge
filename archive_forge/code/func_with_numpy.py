import os
import gc
import sys
from joblib._multiprocessing_helpers import mp
from joblib.testing import SkipTest, skipif
def with_numpy(func):
    """A decorator to skip tests requiring numpy."""

    def my_func():
        raise SkipTest('Test requires numpy')
    return my_func