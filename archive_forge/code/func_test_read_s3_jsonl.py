import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.single_cpu
@td.skip_if_not_us_locale
def test_read_s3_jsonl(self, s3_public_bucket_with_data, s3so):
    result = read_json(f's3n://{s3_public_bucket_with_data.name}/items.jsonl', lines=True, storage_options=s3so)
    expected = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)