from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_type_comparisons():
    val = pa.int32()
    assert val == pa.int32()
    assert val == 'int32'
    assert val != 5