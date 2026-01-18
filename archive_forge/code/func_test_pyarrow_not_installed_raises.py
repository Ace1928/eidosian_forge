import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
@td.skip_if_installed('pyarrow')
def test_pyarrow_not_installed_raises():
    msg = re.escape('pyarrow>=10.0.1 is required for PyArrow backed')
    with pytest.raises(ImportError, match=msg):
        StringDtype(storage='pyarrow')
    with pytest.raises(ImportError, match=msg):
        ArrowStringArray([])
    with pytest.raises(ImportError, match=msg):
        ArrowStringArrayNumpySemantics([])
    with pytest.raises(ImportError, match=msg):
        ArrowStringArray._from_sequence(['a', None, 'b'])