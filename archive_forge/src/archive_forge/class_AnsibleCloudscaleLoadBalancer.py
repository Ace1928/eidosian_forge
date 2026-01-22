from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
class AnsibleCloudscaleLoadBalancer(AnsibleCloudscaleBase):

    def __init__(self, module):
        super(AnsibleCloudscaleLoadBalancer, self).__init__(module, resource_name='load-balancers', resource_create_param_keys=['name', 'flavor', 'zone', 'vip_addresses', 'tags'], resource_update_param_keys=['name', 'tags'])

    def create(self, resource, data=None):
        super().create(resource)
        if not self._module.check_mode:
            resource = self.wait_for_state('status', ('running',))
        return resource