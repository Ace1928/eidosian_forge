import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.mark.pandas
def test_struct_w_ext_array_to_pandas(struct_w_ext_data):
    import pandas as pd
    result = struct_w_ext_data[0].to_pandas()
    expected = pd.Series([{'f0': 1}, {'f0': 2}, {'f0': 3}], dtype=object)
    pd.testing.assert_series_equal(result, expected)
    result = struct_w_ext_data[1].to_pandas()
    expected = pd.Series([{'f1': b'123'}, {'f1': b'456'}, {'f1': b'789'}], dtype=object)
    pd.testing.assert_series_equal(result, expected)