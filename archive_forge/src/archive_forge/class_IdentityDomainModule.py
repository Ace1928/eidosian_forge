from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from ansible_collections.openstack.cloud.plugins.module_utils.resource import StateMachine
class IdentityDomainModule(OpenStackModule):
    argument_spec = dict(description=dict(), is_enabled=dict(type='bool', aliases=['enabled']), name=dict(required=True), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(supports_check_mode=True)

    class _StateMachine(StateMachine):

        def _delete(self, resource, attributes, timeout, wait, **kwargs):
            self.connection.delete_domain(resource['id'])

    def run(self):
        sm = self._StateMachine(connection=self.conn, service_name='identity', type_name='domain', sdk=self.sdk)
        kwargs = dict(((k, self.params[k]) for k in ['state', 'timeout'] if self.params[k] is not None))
        kwargs['attributes'] = dict(((k, self.params[k]) for k in ['description', 'is_enabled', 'name'] if self.params[k] is not None))
        domain, is_changed = sm(check_mode=self.ansible.check_mode, updateable_attributes=None, non_updateable_attributes=None, wait=False, **kwargs)
        if domain is None:
            self.exit_json(changed=is_changed)
        else:
            self.exit_json(changed=is_changed, domain=domain.to_dict(computed=False))