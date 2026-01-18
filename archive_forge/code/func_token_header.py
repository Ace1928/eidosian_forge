from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.common.parameters import env_fallback
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
@property
def token_header(self):
    return self.resp.headers.get('X-Auth-Token')