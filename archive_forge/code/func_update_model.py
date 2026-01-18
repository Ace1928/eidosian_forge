from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import quote
import json
import re
import xml.etree.ElementTree as ET
def update_model(self, model_handle, attrs):
    """
        Update a model's attributes
        :param model_handle: The model's handle ID
        :type model_handle: str
        :param attrs: Model's attributes to update. {'<name/id>': '<attr>'}
        :type attrs: dict
        :returns: Nothing; exits on error or updates self.results
        :rtype: None
        """
    update_url = self.build_url('/model/%s?' % model_handle)
    for name, val in list(attrs.items()):
        if val is None:
            val = ''
        val = self.urlencode(str(val))
        if not update_url.endswith('?'):
            update_url += '&'
        update_url += 'attr=%s&val=%s' % (self.attr_id(name) or name, val)
    resp, info = fetch_url(self.module, update_url, method='PUT', headers={'Content-Type': 'application/json', 'Accept': 'application/json'}, use_proxy=self.module.params['use_proxy'])
    status_code = info['status']
    if status_code >= 400:
        body = info['body']
    else:
        body = '' if resp is None else resp.read()
    if status_code != 200:
        self.result['msg'] = 'HTTP PUT error %s: %s: %s' % (status_code, update_url, body)
        self.module.fail_json(**self.result)
    json_resp = json.loads(body)
    '\n        Example success response:\n        {\'model-update-response-list\':{\'model-responses\':{\'model\':{\'@error\':\'Success\',\'@mh\':\'0x1010e76\',\'attribute\':{\'@error\':\'Success\',\'@id\':\'0x1295d\'}}}}}"\n        Example failure response:\n        {\'model-update-response-list\': {\'model-responses\': {\'model\': {\'@error\': \'PartialFailure\', \'@mh\': \'0x1010e76\', \'attribute\': {\'@error-message\': \'brn0vlappua001: You do not have permission to set attribute Network_Address for this model.\', \'@error\': \'Error\', \'@id\': \'0x12d7f\'}}}}}\n        '
    model_resp = json_resp['model-update-response-list']['model-responses']['model']
    if model_resp['@error'] != 'Success':
        self.result['msg'] = str(model_resp['attribute'])
        self.module.fail_json(**self.result)
    self.result['msg'] = self.success_msg
    self.result['changed_attrs'].update(attrs)
    self.result['changed'] = True