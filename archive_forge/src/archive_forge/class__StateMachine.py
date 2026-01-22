from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from ansible_collections.openstack.cloud.plugins.module_utils.resource import StateMachine
class _StateMachine(StateMachine):

    def _find(self, attributes, **kwargs):
        kwargs = dict(((k, attributes[k]) for k in ['domain_id'] if k in attributes and attributes[k] is not None))
        return self.find_function(attributes['name'], **kwargs)