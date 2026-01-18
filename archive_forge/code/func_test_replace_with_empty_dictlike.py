from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_with_empty_dictlike(self, mix_abc):
    df = DataFrame(mix_abc)
    tm.assert_frame_equal(df, df.replace({}))
    tm.assert_frame_equal(df, df.replace(Series([], dtype=object)))
    tm.assert_frame_equal(df, df.replace({'b': {}}))
    tm.assert_frame_equal(df, df.replace(Series({'b': {}})))