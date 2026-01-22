import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
@pytest.fixture(params=[np.array(['a', 'b'], dtype=object), np.array([0, 1], dtype=float), np.array([0, 1], dtype=int), np.array([0, 1 + 2j], dtype=complex), np.array([True, False], dtype=bool), np.array([0, 1], dtype='datetime64[ns]'), np.array([0, 1], dtype='timedelta64[ns]')])
def any_numpy_array(request):
    """
    Parametrized fixture for NumPy arrays with different dtypes.

    This excludes string and bytes.
    """
    return request.param