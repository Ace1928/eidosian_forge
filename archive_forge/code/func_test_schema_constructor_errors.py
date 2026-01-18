from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_constructor_errors():
    msg = "Do not call Schema's constructor directly, use `pyarrow.schema` instead"
    with pytest.raises(TypeError, match=msg):
        pa.Schema()