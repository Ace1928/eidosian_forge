import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_validate_reduction_keyword_args():
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    msg = "the 'keepdims' parameter is not supported .*all"
    with pytest.raises(ValueError, match=msg):
        arr.all(keepdims=True)