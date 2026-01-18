import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_optional_from_23seq(self):
    obj = self.num23seq
    shape = (len(obj), len(obj[0]))
    a = self.array(shape, intent.optional, obj)
    assert a.arr.shape == shape
    assert not a.has_shared_memory()
    a = self.array(shape, intent.optional.c, obj)
    assert a.arr.shape == shape
    assert not a.has_shared_memory()