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
Check if the value is set in REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE or OMAM_CA_BUNDLE or returns None