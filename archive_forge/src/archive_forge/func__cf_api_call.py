from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
def _cf_api_call(self, api_call, method='GET', payload=None):
    result, status = self._cf_simple_api_call(api_call, method, payload)
    data = result['result']
    if 'result_info' in result:
        pagination = result['result_info']
        if pagination['total_pages'] > 1:
            next_page = int(pagination['page']) + 1
            parameters = ['page={0}'.format(next_page)]
            if '?' in api_call:
                raw_api_call, query = api_call.split('?', 1)
                parameters += [param for param in query.split('&') if not param.startswith('page')]
            else:
                raw_api_call = api_call
            while next_page <= pagination['total_pages']:
                raw_api_call += '?' + '&'.join(parameters)
                result, status = self._cf_simple_api_call(raw_api_call, method, payload)
                data += result['result']
                next_page += 1
    return (data, status)