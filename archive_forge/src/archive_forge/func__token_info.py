import json
import os
from google.auth import _helpers
import google.auth.transport.requests
import google.auth.transport.urllib3
import pytest
import requests
import urllib3
def _token_info(access_token=None, id_token=None):
    query_params = {}
    if access_token is not None:
        query_params['access_token'] = access_token
    elif id_token is not None:
        query_params['id_token'] = id_token
    else:
        raise ValueError('No token specified.')
    url = _helpers.update_query(TOKEN_INFO_URL, query_params)
    response = http_request(url=url, method='GET')
    return json.loads(response.data.decode('utf-8'))