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
@pytest.mark.parametrize('responder, read_method', [(AllHeaderCSVResponder, pd.read_csv)], indirect=['responder'])
def test_server_and_all_custom_headers(responder, read_method):
    custom_user_agent = 'Super Cool One'
    custom_auth_token = 'Super Secret One'
    storage_options = {'User-Agent': custom_user_agent, 'Auth': custom_auth_token}
    read_method = wait_until_ready(read_method)
    df_http = read_method(f'http://localhost:{responder}', storage_options=storage_options)
    df_http = df_http[df_http['0'].isin(storage_options.keys())]
    df_http = df_http.sort_values(['0']).reset_index()
    df_http = df_http[['0', '1']]
    keys = list(storage_options.keys())
    df_true = pd.DataFrame({'0': keys, '1': [storage_options[k] for k in keys]})
    df_true = df_true.sort_values(['0'])
    df_true = df_true.reset_index().drop(['index'], axis=1)
    tm.assert_frame_equal(df_true, df_http)