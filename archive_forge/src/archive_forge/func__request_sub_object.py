from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def _request_sub_object(self, sub_obj):
    directive_state = self._module.params['member_state']
    if directive_state not in ['present', 'absent']:
        raise AssertionError('Not invalid member_state directive.')
    status = None
    result_data = None
    if directive_state == 'present':
        status, result_data = self._conn.send_request(url=sub_obj['get'], params=None, method='GET')
        if status == 200:
            status, result_data = self._conn.send_request(url=sub_obj['put'], data=json.dumps(sub_obj['put_payload']), method='PUT')
            if status == 405:
                status, result_data = self._conn.send_request(url=sub_obj['post'], data=json.dumps(sub_obj['post_payload']), method='POST')
        else:
            status, result_data = self._conn.send_request(url=sub_obj['post'], data=json.dumps(sub_obj['post_payload']), method='POST')
    else:
        status, result_data = self._conn.send_request(url=sub_obj['delete'], params=None, method='DELETE')
    result_data = self.formatresponse(result_data, status, vdom=sub_obj['vdom'])
    return result_data