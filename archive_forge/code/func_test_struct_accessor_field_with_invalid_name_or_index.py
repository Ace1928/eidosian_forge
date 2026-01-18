import re
import pytest
from pandas.compat.pyarrow import (
from pandas import (
import pandas._testing as tm
def test_struct_accessor_field_with_invalid_name_or_index():
    ser = Series([], dtype=ArrowDtype(pa.struct([('field', pa.int64())])))
    with pytest.raises(ValueError, match='name_or_index must be an int, str,'):
        ser.struct.field(1.1)