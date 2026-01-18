import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_raising_func(raising_func_fixture):
    _, func_name = raising_func_fixture
    with pytest.raises(MyError, match='error raised by scalar UDF'):
        pc.call_function(func_name, [], length=1)