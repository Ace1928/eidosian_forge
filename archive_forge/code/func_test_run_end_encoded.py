import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_run_end_encoded():
    run_ends = [3, 5, 10, 12, 19]
    values = [1, 2, 1, None, 3]
    arr = pa.RunEndEncodedArray.from_arrays(run_ends, values)
    scalar = arr[0]
    assert isinstance(scalar, pa.RunEndEncodedScalar)
    assert isinstance(scalar.value, pa.Int64Scalar)
    assert scalar.value == pa.array(values)[0]
    assert scalar.as_py() == 1
    scalar = arr[10]
    assert isinstance(scalar.value, pa.Int64Scalar)
    assert scalar.as_py() is None
    with pytest.raises(NotImplementedError):
        pa.scalar(1, pa.run_end_encoded(pa.int64(), pa.int64()))