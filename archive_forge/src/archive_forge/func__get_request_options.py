import base64
import copy
import logging
import os
from oslo_utils import importutils
import requests
def _get_request_options(self, method, headers):
    headers = self._update_headers(headers)
    if method in ['post', 'put']:
        content_type = headers.get('content-type', 'application/json')
        headers['content-type'] = content_type
    options = copy.deepcopy(self.ssl_options)
    options['headers'] = headers
    return options