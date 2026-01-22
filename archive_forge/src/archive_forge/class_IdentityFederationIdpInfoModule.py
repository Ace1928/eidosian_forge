from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityFederationIdpInfoModule(OpenStackModule):
    argument_spec = dict(id=dict(aliases=['name']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = dict(((k, self.params[k]) for k in ['id'] if self.params[k] is not None))
        identity_providers = self.conn.identity.identity_providers(**kwargs)
        self.exit_json(changed=False, identity_providers=[i.to_dict(computed=False) for i in identity_providers])