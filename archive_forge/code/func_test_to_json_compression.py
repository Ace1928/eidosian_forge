from io import (
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('to_infer', [True, False])
@pytest.mark.parametrize('read_infer', [True, False])
def test_to_json_compression(compression_only, read_infer, to_infer, compression_to_extension, infer_string):
    with pd.option_context('future.infer_string', infer_string):
        compression = compression_only
        filename = 'test.'
        filename += compression_to_extension[compression]
        df = pd.DataFrame({'A': [1]})
        to_compression = 'infer' if to_infer else compression
        read_compression = 'infer' if read_infer else compression
        with tm.ensure_clean(filename) as path:
            df.to_json(path, compression=to_compression)
            result = pd.read_json(path, compression=read_compression)
            tm.assert_frame_equal(result, df)