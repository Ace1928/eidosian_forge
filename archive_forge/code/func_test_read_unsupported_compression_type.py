from io import (
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def test_read_unsupported_compression_type():
    with tm.ensure_clean() as path:
        msg = 'Unrecognized compression type: unsupported'
        with pytest.raises(ValueError, match=msg):
            pd.read_json(path, compression='unsupported')