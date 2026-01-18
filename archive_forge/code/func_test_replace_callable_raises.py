from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('repl', [lambda: None, lambda m, x: None, lambda m, x, y=None: None])
def test_replace_callable_raises(any_string_dtype, repl):
    values = Series(['fooBAD__barBAD', np.nan], dtype=any_string_dtype)
    msg = '((takes)|(missing)) (?(2)from \\d+ to )?\\d+ (?(3)required )positional arguments?'
    with pytest.raises(TypeError, match=msg):
        with tm.maybe_produces_warning(PerformanceWarning, using_pyarrow(any_string_dtype)):
            values.str.replace('a', repl, regex=True)