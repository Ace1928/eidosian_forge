from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def __to_local(self, data, http_status, is_array=False):
    try:
        resp = json.loads(data)
    except Exception:
        resp = {'raw': data}
    if is_array and type(resp) is not list:
        resp = [resp]
    if is_array and 'http_status' not in resp[0]:
        resp[0]['http_status'] = http_status
    elif not is_array and 'status' not in resp:
        resp['http_status'] = http_status
    return resp