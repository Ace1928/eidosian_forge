from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_type_timestamp_with_tz():
    tz = 'America/Los_Angeles'
    t = pa.timestamp('ns', tz=tz)
    assert t.unit == 'ns'
    assert t.tz == tz