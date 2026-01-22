from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from ansible_collections.openstack.cloud.plugins.module_utils.resource import StateMachine
class IdentityGroupModule(OpenStackModule):
    argument_spec = dict(description=dict(), domain_id=dict(), name=dict(required=True), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(supports_check_mode=True)

    class _StateMachine(StateMachine):

        def _find(self, attributes, **kwargs):
            kwargs = dict(((k, attributes[k]) for k in ['domain_id'] if k in attributes and attributes[k] is not None))
            return self.find_function(attributes['name'], **kwargs)

    def run(self):
        sm = self._StateMachine(connection=self.conn, service_name='identity', type_name='group', sdk=self.sdk)
        kwargs = dict(((k, self.params[k]) for k in ['state', 'timeout'] if self.params[k] is not None))
        kwargs['attributes'] = dict(((k, self.params[k]) for k in ['description', 'domain_id', 'name'] if self.params[k] is not None))
        group, is_changed = sm(check_mode=self.ansible.check_mode, updateable_attributes=None, non_updateable_attributes=['domain_id'], wait=False, **kwargs)
        if group is None:
            self.exit_json(changed=is_changed)
        else:
            self.exit_json(changed=is_changed, group=group.to_dict(computed=False))