from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOVPCInfo(object):

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.module.params.pop('oauth_token')
        self.name = self.module.params.pop('name', '')
        self.members = self.module.params.pop('members', False)

    def get_by_name(self):
        page = 1
        while page is not None:
            response = self.rest.get('vpcs?page={0}'.format(page))
            json_data = response.json
            if response.status_code == 200:
                for vpc in json_data['vpcs']:
                    if vpc.get('name', None) == self.name:
                        return vpc
                if 'links' in json_data and 'pages' in json_data['links'] and ('next' in json_data['links']['pages']):
                    page += 1
                else:
                    page = None
        return None

    def get(self):
        if self.module.check_mode:
            return self.module.exit_json(changed=False)
        if not self.members:
            base_url = 'vpcs?'
            vpcs = self.rest.get_paginated_data(base_url=base_url, data_key_name='vpcs')
            self.module.exit_json(changed=False, data=vpcs)
        else:
            vpc = self.get_by_name()
            if vpc is not None:
                vpc_id = vpc.get('id', None)
                if vpc_id is not None:
                    response = self.rest.get('vpcs/{0}/members'.format(vpc_id))
                    json = response.json
                    if response.status_code != 200:
                        self.module.fail_json(msg='Failed to find VPC named {0}: {1}'.format(self.name, json['message']))
                    else:
                        self.module.exit_json(changed=False, data=json)
                else:
                    self.module.fail_json(changed=False, msg='Unexpected error, please file a bug')
            else:
                self.module.fail_json(changed=False, msg='Could not find a VPC named {0}'.format(self.name))