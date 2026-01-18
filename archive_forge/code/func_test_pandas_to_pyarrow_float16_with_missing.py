from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.pandas
def test_pandas_to_pyarrow_float16_with_missing():
    if Version(pd.__version__) < Version('1.5.0'):
        pytest.skip('__dataframe__ added to pandas in 1.5.0')
    np_array = np.array([0, np.nan, 2], dtype=np.float16)
    df = pd.DataFrame({'a': np_array})
    with pytest.raises(NotImplementedError):
        pi.from_dataframe(df)