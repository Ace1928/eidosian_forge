import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase

        Test issue https://github.com/numba/numba/issues/3355
        