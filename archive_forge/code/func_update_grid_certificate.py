from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def update_grid_certificate(self, cert_type):
    api = 'api/v3/grid/%s/update' % cert_type
    response, error = self.rest_api.post(api, self.data)
    if error:
        self.module.fail_json(msg=error)