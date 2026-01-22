from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
def _gandi_api_call(self, api_call, method='GET', payload=None, error_on_404=True):
    headers = {'Authorization': 'Apikey {0}'.format(self.api_key), 'Content-Type': 'application/json'}
    data = None
    if payload:
        try:
            data = json.dumps(payload)
        except Exception as e:
            self.module.fail_json(msg='Failed to encode payload as JSON: %s ' % to_native(e))
    resp, info = fetch_url(self.module, self.api_endpoint + api_call, headers=headers, data=data, method=method)
    error_msg = ''
    if info['status'] >= 400 and (info['status'] != 404 or error_on_404):
        err_s = self.error_strings.get(info['status'], '')
        error_msg = 'API Error {0}: {1}'.format(err_s, self._build_error_message(self.module, info))
    result = None
    try:
        content = resp.read()
    except AttributeError:
        content = None
    if content:
        try:
            result = json.loads(to_text(content, errors='surrogate_or_strict'))
        except getattr(json, 'JSONDecodeError', ValueError) as e:
            error_msg += '; Failed to parse API response with error {0}: {1}'.format(to_native(e), content)
    if error_msg:
        self.module.fail_json(msg=error_msg)
    return (result, info['status'])