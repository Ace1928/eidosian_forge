from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.common.parameters import env_fallback
Check if the value is set in REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE or OMAM_CA_BUNDLE or True as ssl has to
        be validated from omsdk with single param and is default to false in omsdk