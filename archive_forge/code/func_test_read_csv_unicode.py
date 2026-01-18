from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_read_csv_unicode(all_parsers):
    parser = all_parsers
    data = BytesIO('Łaski, Jan;1'.encode())
    result = parser.read_csv(data, sep=';', encoding='utf-8', header=None)
    expected = DataFrame([['Łaski, Jan', 1]])
    tm.assert_frame_equal(result, expected)