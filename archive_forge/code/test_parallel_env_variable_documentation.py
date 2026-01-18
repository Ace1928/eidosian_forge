from numba.np.ufunc.parallel import get_thread_count
from os import environ as env
from numba.core import config
import unittest

        Tests the NUMBA_NUM_THREADS env variable behaves as expected.
        