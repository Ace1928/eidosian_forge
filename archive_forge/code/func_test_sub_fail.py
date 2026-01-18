import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_sub_fail(self, using_infer_string):
    index = pd.Index([str(i) for i in range(10)])
    if using_infer_string:
        import pyarrow as pa
        err = pa.lib.ArrowNotImplementedError
        msg = 'has no kernel'
    else:
        err = TypeError
        msg = 'unsupported operand type|Cannot broadcast'
    with pytest.raises(err, match=msg):
        index - 'a'
    with pytest.raises(err, match=msg):
        index - index
    with pytest.raises(err, match=msg):
        index - index.tolist()
    with pytest.raises(err, match=msg):
        index.tolist() - index