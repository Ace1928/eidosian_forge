from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.parameters import env_fallback
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def _url_common_args_spec(self, method, api_timeout, headers=None):
    """Creates an argument common spec"""
    req_header = self._headers
    if headers:
        req_header.update(headers)
    if api_timeout is None:
        api_timeout = self.timeout
    if self.ca_path is None:
        self.ca_path = self._get_omam_ca_env()
    url_kwargs = {'method': method, 'validate_certs': self.validate_certs, 'ca_path': self.ca_path, 'use_proxy': self.use_proxy, 'headers': req_header, 'timeout': api_timeout, 'follow_redirects': 'all'}
    return url_kwargs