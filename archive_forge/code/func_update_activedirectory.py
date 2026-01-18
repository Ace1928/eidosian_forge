from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def update_activedirectory(self, activedirectory_id, updated_activedirectory):
    api = 'Storage/ActiveDirectory/' + activedirectory_id
    data = {'region': self.parameters['region'], 'DNS': updated_activedirectory['DNS'], 'domain': updated_activedirectory['domain'], 'username': updated_activedirectory['username'], 'password': updated_activedirectory['password'], 'netBIOS': updated_activedirectory['netBIOS']}
    response, error = self.rest_api.put(api, data)
    if not error:
        return response
    else:
        self.module.fail_json(msg=response['message'])