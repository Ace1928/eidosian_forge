from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
@pytest.mark.pandas
def test_pandas_serialize_round_trip_not_string_columns():
    df = pd.DataFrame(list(zip([1.5, 1.6, 1.7], 'abc')))
    buf = pa.serialize_pandas(df)
    result = pa.deserialize_pandas(buf)
    assert_frame_equal(result, df)