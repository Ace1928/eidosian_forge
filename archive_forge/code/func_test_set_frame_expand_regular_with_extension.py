import itertools
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_set_frame_expand_regular_with_extension(self, data):
    df = pd.DataFrame({'A': [1] * len(data)})
    df['B'] = data
    expected = pd.DataFrame({'A': [1] * len(data), 'B': data})
    tm.assert_frame_equal(df, expected)