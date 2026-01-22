from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOCDNEndpoint(object):

    def __init__(self, module):
        self.module = module
        self.rest = DigitalOceanHelper(module)
        self.token = self.module.params.pop('oauth_token')

    def get_cdn_endpoints(self):
        cdns = self.rest.get_paginated_data(base_url='cdn/endpoints?', data_key_name='endpoints')
        return cdns

    def get_cdn_endpoint(self):
        cdns = self.rest.get_paginated_data(base_url='cdn/endpoints?', data_key_name='endpoints')
        found = None
        for cdn in cdns:
            if cdn.get('origin') == self.module.params.get('origin'):
                found = cdn
                for key in ['ttl', 'certificate_id']:
                    if self.module.params.get(key) != cdn.get(key):
                        return (found, True)
        return (found, False)

    def create(self):
        cdn, needs_update = self.get_cdn_endpoint()
        if cdn is not None:
            if not needs_update:
                self.module.exit_json(changed=False, msg=cdn)
            if needs_update:
                if self.module.check_mode:
                    self.module.exit_json(changed=True)
                request_params = dict(self.module.params)
                endpoint = 'cdn/endpoints'
                response = self.rest.put('{0}/{1}'.format(endpoint, cdn.get('id')), data=request_params)
                status_code = response.status_code
                json_data = response.json
                if status_code != 200:
                    self.module.fail_json(changed=False, msg='Failed to put {0} information due to error [HTTP {1}: {2}]'.format(endpoint, status_code, json_data.get('message', '(empty error message)')))
                self.module.exit_json(changed=True, data=json_data)
        else:
            if self.module.check_mode:
                self.module.exit_json(changed=True)
            request_params = dict(self.module.params)
            endpoint = 'cdn/endpoints'
            response = self.rest.post(endpoint, data=request_params)
            status_code = response.status_code
            json_data = response.json
            if status_code != 201:
                self.module.fail_json(changed=False, msg='Failed to post {0} information due to error [HTTP {1}: {2}]'.format(endpoint, status_code, json_data.get('message', '(empty error message)')))
            self.module.exit_json(changed=True, data=json_data)

    def delete(self):
        cdn, needs_update = self.get_cdn_endpoint()
        if cdn is not None:
            if self.module.check_mode:
                self.module.exit_json(changed=True)
            endpoint = 'cdn/endpoints/{0}'.format(cdn.get('id'))
            response = self.rest.delete(endpoint)
            status_code = response.status_code
            json_data = response.json
            if status_code != 204:
                self.module.fail_json(changed=False, msg='Failed to delete {0} information due to error [HTTP {1}: {2}]'.format(endpoint, status_code, json_data.get('message', '(empty error message)')))
            self.module.exit_json(changed=True, msg='Deleted CDN Endpoint {0} ({1})'.format(cdn.get('origin'), cdn.get('id')))
        else:
            self.module.exit_json(changed=False)