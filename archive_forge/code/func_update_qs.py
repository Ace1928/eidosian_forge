from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def update_qs(self, params):
    """Append key-value pairs to self.filter_string"""
    accepted_params = dict(((k, v) for k, v in params.items() if v is not None))
    if accepted_params:
        if self.filter_string:
            self.filter_string += '&'
        else:
            self.filter_string = '?'
        self.filter_string += '&'.join(['%s=%s' % (k, v) for k, v in accepted_params.items()])