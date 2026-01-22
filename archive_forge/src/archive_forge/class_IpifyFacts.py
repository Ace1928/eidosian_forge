from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text
class IpifyFacts(object):

    def __init__(self):
        self.api_url = module.params.get('api_url')
        self.timeout = module.params.get('timeout')

    def run(self):
        result = {'ipify_public_ip': None}
        response, info = fetch_url(module=module, url=self.api_url + '?format=json', force=True, timeout=self.timeout)
        if not response:
            module.fail_json(msg='No valid or no response from url %s within %s seconds (timeout)' % (self.api_url, self.timeout))
        data = json.loads(to_text(response.read()))
        result['ipify_public_ip'] = data.get('ip')
        return result