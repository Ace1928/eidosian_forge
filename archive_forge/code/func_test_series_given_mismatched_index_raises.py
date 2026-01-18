import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_series_given_mismatched_index_raises(self, data):
    msg = 'Length of values \\(3\\) does not match length of index \\(5\\)'
    with pytest.raises(ValueError, match=msg):
        pd.Series(data[:3], index=[0, 1, 2, 3, 4])