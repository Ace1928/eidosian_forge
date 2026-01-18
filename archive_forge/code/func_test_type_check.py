import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_type_check(errors):
    df = DataFrame({'a': [1, -3.14, 7], 'b': ['4', '5', '6']})
    kwargs = {'errors': errors} if errors is not None else {}
    with pytest.raises(TypeError, match='1-d array'):
        to_numeric(df, **kwargs)