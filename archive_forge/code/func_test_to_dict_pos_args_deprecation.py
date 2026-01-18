from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_to_dict_pos_args_deprecation(self):
    df = DataFrame({'a': [1, 2, 3]})
    msg = "Starting with pandas version 3.0 all arguments of to_dict except for the argument 'orient' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.to_dict('records', {})