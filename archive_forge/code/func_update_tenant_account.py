from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import (
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import (
def update_tenant_account(self, account_id):
    api = 'api/v3/grid/accounts/' + account_id
    if 'password' in self.data:
        del self.data['password']
    if 'grantRootAccessToGroup' in self.data:
        del self.data['grantRootAccessToGroup']
    response, error = self.rest_api.put(api, self.data)
    if error:
        self.module.fail_json(msg=error)
    return response['data']