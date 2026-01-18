from numba import cuda
import numpy as np
from numba.cuda.testing import CUDATestCase
from numba.tests.support import override_config
import unittest

    Tests the jit decorator with no types provided.
    