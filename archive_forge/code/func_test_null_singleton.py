import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_null_singleton():
    with pytest.raises(RuntimeError):
        pa.NullScalar()