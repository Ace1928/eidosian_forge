import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_pos_args_deprecation(self):
    df = DataFrame({'name': ['Raphael', 'Donatello'], 'age': [26, 45], 'height': [181.23, 177.65]})
    msg = "Starting with pandas version 3.0 all arguments of to_latex except for the argument 'buf' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.to_latex(None, None)