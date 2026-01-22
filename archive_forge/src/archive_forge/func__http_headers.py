import os
import requests
import testtools
from keystone.tests.common import auth as common_auth
def _http_headers(self, token=None):
    headers = {'content-type': 'application/json'}
    if token:
        headers['X-Auth-Token'] = token
    return headers