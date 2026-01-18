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
def test_one_arg(self):
    MyArray, implements = _new_duck_type_and_implements()

    @implements(dispatched_one_arg)
    def _(array):
        return 'myarray'
    assert_equal(dispatched_one_arg(1), 'original')
    assert_equal(dispatched_one_arg(MyArray()), 'myarray')