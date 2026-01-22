from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from ansible_collections.openstack.cloud.plugins.module_utils.resource import StateMachine
class IdentityProviderModule(OpenStackModule):
    argument_spec = dict(description=dict(), domain_id=dict(), id=dict(required=True, aliases=['name']), is_enabled=dict(type='bool', aliases=['enabled']), remote_ids=dict(type='list', elements='str'), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        sm = StateMachine(connection=self.conn, service_name='identity', type_name='identity_provider', sdk=self.sdk)
        kwargs = dict(((k, self.params[k]) for k in ['state', 'timeout'] if self.params[k] is not None))
        kwargs['attributes'] = dict(((k, self.params[k]) for k in ['description', 'domain_id', 'id', 'is_enabled', 'remote_ids'] if self.params[k] is not None))
        identity_provider, is_changed = sm(check_mode=self.ansible.check_mode, updateable_attributes=None, non_updateable_attributes=['domain_id'], wait=False, **kwargs)
        if identity_provider is None:
            self.exit_json(changed=is_changed)
        else:
            self.exit_json(changed=is_changed, identity_provider=identity_provider.to_dict(computed=False))