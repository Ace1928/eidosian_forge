import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
def test_string_dtype_error_message():
    pytest.importorskip('pyarrow')
    msg = "Storage must be 'python', 'pyarrow' or 'pyarrow_numpy'."
    with pytest.raises(ValueError, match=msg):
        StringDtype('bla')