from numba import cuda, float32, int32
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import ignore_internal_warnings
import re
import unittest
import warnings
def _loc_directive_regex(self):
    pat = '\\.loc\\s+[0-9]+\\s+[0-9]+\\s+[0-9]+'
    return re.compile(pat)