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
@pytest.mark.parametrize('compression', ['', '.gz', '.bz2', '.tar'])
def test_read_json_with_very_long_file_path(self, compression):
    long_json_path = f'{'a' * 1000}.json{compression}'
    with pytest.raises(FileNotFoundError, match=f'File {long_json_path} does not exist'):
        read_json(long_json_path)