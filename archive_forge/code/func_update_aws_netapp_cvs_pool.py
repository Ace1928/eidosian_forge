from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def update_aws_netapp_cvs_pool(self, update_pool_info, pool_id):
    """
        Update a pool
        """
    api = 'Pools/' + pool_id
    pool = {'name': update_pool_info['name'], 'region': self.parameters['region'], 'serviceLevel': update_pool_info['serviceLevel'], 'sizeInBytes': update_pool_info['sizeInBytes'], 'vendorID': update_pool_info['vendorID']}
    dummy, error = self.rest_api.put(api, pool)
    if error is not None:
        self.module.fail_json(changed=False, msg=error)