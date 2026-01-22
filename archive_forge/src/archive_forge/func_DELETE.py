from __future__ import absolute_import
import io
import json
import ssl
import certifi
import logging
import re
from six import PY3
from six.moves.urllib.parse import urlencode
def DELETE(self, url, headers=None, query_params=None, body=None, _preload_content=True, _request_timeout=None):
    return self.request('DELETE', url, headers=headers, query_params=query_params, _preload_content=_preload_content, _request_timeout=_request_timeout, body=body)