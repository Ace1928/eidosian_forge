from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
@pytest.mark.parametrize('margin_name', ['foo', 'one', 666, None, ['a', 'b']])
def test_pivot_table_with_margins_set_margin_name(self, margin_name, data):
    msg = f'Conflicting name "{margin_name}" in margins|margins_name argument must be a string'
    with pytest.raises(ValueError, match=msg):
        pivot_table(data, values='D', index=['A', 'B'], columns=['C'], margins=True, margins_name=margin_name)
    with pytest.raises(ValueError, match=msg):
        pivot_table(data, values='D', index=['C'], columns=['A', 'B'], margins=True, margins_name=margin_name)
    with pytest.raises(ValueError, match=msg):
        pivot_table(data, values='D', index=['A'], columns=['B'], margins=True, margins_name=margin_name)