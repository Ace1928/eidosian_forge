from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_type_integers():
    dtypes = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
    for name in dtypes:
        factory = getattr(pa, name)
        t = factory()
        assert str(t) == name