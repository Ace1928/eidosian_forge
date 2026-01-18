import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_optional_from_2seq(self):
    obj = self.num2seq
    shape = (len(obj),)
    a = self.array(shape, intent.optional, obj)
    assert a.arr.shape == shape
    assert not a.has_shared_memory()