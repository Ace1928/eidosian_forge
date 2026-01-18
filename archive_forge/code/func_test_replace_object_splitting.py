from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_object_splitting(self, using_infer_string):
    df = DataFrame({'a': ['a'], 'b': 'b'})
    if using_infer_string:
        assert len(df._mgr.blocks) == 2
    else:
        assert len(df._mgr.blocks) == 1
    df.replace(to_replace='^\\s*$', value='', inplace=True, regex=True)
    if using_infer_string:
        assert len(df._mgr.blocks) == 2
    else:
        assert len(df._mgr.blocks) == 1