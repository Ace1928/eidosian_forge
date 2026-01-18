import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_wrong_num_labels():
    msg = 'Bin labels must be one fewer than the number of bin edges'
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]
    with pytest.raises(ValueError, match=msg):
        cut(data, [0, 1, 10], labels=['foo', 'bar', 'baz'])