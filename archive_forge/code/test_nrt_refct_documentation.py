import gc
import numpy as np
import unittest
from numba import njit
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin

        Test issue #1734
        