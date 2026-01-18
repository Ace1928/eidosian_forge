from io import (
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def test_write_unsupported_compression_type():
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
    with tm.ensure_clean() as path:
        msg = 'Unrecognized compression type: unsupported'
        with pytest.raises(ValueError, match=msg):
            df.to_json(path, compression='unsupported')