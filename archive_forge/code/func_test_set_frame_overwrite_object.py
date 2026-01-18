import itertools
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_set_frame_overwrite_object(self, data):
    df = pd.DataFrame({'A': [1] * len(data)}, dtype=object)
    df['A'] = data
    assert df.dtypes['A'] == data.dtype