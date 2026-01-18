import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask

    Test conversion from pyarrow array to numpy array.

    Modifies the pyarrow buffer to contain padding and offset, which are
    considered valid buffers by pyarrow.

    Also tests empty pyarrow arrays with non empty buffers.
    See https://github.com/pandas-dev/pandas/issues/40896
    