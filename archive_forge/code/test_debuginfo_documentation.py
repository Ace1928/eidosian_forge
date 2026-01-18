from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest

    These tests only checks the compiled PTX for debuginfo section
    