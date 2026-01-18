from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_equals_invalid_type():
    schema = pa.schema([pa.field('a', pa.int64())])
    for val in [None, 'string', pa.array([1, 2])]:
        with pytest.raises(TypeError):
            schema.equals(val)