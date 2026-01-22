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
def _args_with_session(self, method, api_timeout, headers=None):
    """Creates an argument spec, in case of authentication with session"""
    url_kwargs = self._url_common_args_spec(method, api_timeout, headers=headers)
    url_kwargs['force_basic_auth'] = False
    return url_kwargs