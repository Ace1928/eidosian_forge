import gzip
import http.server
from io import BytesIO
import multiprocessing
import socket
import time
import urllib.error
import pytest
from pandas.compat import is_ci_environment
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('responder, read_method, parquet_engine', [(CSVUserAgentResponder, pd.read_csv, None), (JSONUserAgentResponder, pd.read_json, None), (ParquetPyArrowUserAgentResponder, pd.read_parquet, 'pyarrow'), pytest.param(ParquetFastParquetUserAgentResponder, pd.read_parquet, 'fastparquet', marks=[td.skip_array_manager_not_yet_implemented]), (PickleUserAgentResponder, pd.read_pickle, None), (StataUserAgentResponder, pd.read_stata, None), (GzippedCSVUserAgentResponder, pd.read_csv, None), (GzippedJSONUserAgentResponder, pd.read_json, None)], indirect=['responder'])
def test_server_and_default_headers(responder, read_method, parquet_engine):
    if parquet_engine is not None:
        pytest.importorskip(parquet_engine)
        if parquet_engine == 'fastparquet':
            pytest.importorskip('fsspec')
    read_method = wait_until_ready(read_method)
    if parquet_engine is None:
        df_http = read_method(f'http://localhost:{responder}')
    else:
        df_http = read_method(f'http://localhost:{responder}', engine=parquet_engine)
    assert not df_http.empty