from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def update_grid_gateway_server(self, gateway_id):
    api = 'api/v3/private/gateway-configs/%s/server-config' % gateway_id
    response, error = self.rest_api.put(api, self.data_server)
    if error:
        self.module.fail_json(msg=error)
    return response['data']