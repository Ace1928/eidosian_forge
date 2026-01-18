import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_override_sum(self):
    MyArray, implements = _new_duck_type_and_implements()

    @implements(np.sum)
    def _(array):
        return 'yes'
    assert_equal(np.sum(MyArray()), 'yes')