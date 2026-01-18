from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
def test_string_validity_buffer_no_missing() -> None:
    pytest.importorskip('pyarrow', '11.0.0')
    df = pd.DataFrame({'a': ['x', None]}, dtype='large_string[pyarrow]')
    validity = df.__dataframe__().get_column_by_name('a').get_buffers()['validity']
    assert validity is not None
    result = validity[1]
    expected = (DtypeKind.BOOL, 1, ArrowCTypes.BOOL, '=')
    assert result == expected