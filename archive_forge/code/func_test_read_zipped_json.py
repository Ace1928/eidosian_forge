from io import (
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def test_read_zipped_json(datapath):
    uncompressed_path = datapath('io', 'json', 'data', 'tsframe_v012.json')
    uncompressed_df = pd.read_json(uncompressed_path)
    compressed_path = datapath('io', 'json', 'data', 'tsframe_v012.json.zip')
    compressed_df = pd.read_json(compressed_path, compression='zip')
    tm.assert_frame_equal(uncompressed_df, compressed_df)