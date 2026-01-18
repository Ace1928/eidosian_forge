from __future__ import absolute_import, division, print_function
import base64
from .vultr_v2 import AnsibleVultr
def transform_resource(self, resource):
    if not resource:
        return resource
    features = resource.get('features', list())
    if 'backups' in self.module.params:
        resource['backups'] = 'enabled' if 'auto_backups' in features else 'disabled'
    if 'ddos_protection' in self.module.params:
        resource['ddos_protection'] = 'ddos_protection' in features
    if 'persistent_pxe' in self.module.params:
        resource['persistent_pxe'] = 'persistent_pxe' in features
    resource['enable_ipv6'] = 'ipv6' in features
    if 'vpcs' in self.module.params:
        resource['vpcs'] = self.get_resource_vpcs(resource=resource)
    if 'vpc2s' in self.module.params:
        resource['vpc2s'] = self.get_resource_vpcs(resource=resource, api_version='v2')
    return resource