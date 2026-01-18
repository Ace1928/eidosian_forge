from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
def test_to_clipboard_pos_args_deprecation(self):
    df = DataFrame({'a': [1, 2, 3]})
    msg = 'Starting with pandas version 3.0 all arguments of to_clipboard will be keyword-only.'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.to_clipboard(True, None)