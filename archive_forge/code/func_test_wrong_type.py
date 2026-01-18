import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
def test_wrong_type(self):
    with pytest.raises(TypeError):
        self.conv({})
    with pytest.raises(TypeError):
        self.conv([])