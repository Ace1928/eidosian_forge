import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_use_output_signature_for_all_arguments(self):
    res = np.power(1.5, 2.8, dtype=np.intp, casting='unsafe')
    assert res == 1
    res = np.power(1.5, 2.8, signature=(None, None, np.intp), casting='unsafe')
    assert res == 1
    with pytest.raises(TypeError):
        np.power(1.5, 2.8, dtype=np.intp)