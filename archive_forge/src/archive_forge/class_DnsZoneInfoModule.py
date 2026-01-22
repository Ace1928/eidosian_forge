from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class DnsZoneInfoModule(OpenStackModule):
    argument_spec = dict(description=dict(), email=dict(), name=dict(), ttl=dict(type='int'), type=dict(choices=['primary', 'secondary']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = dict(((k, self.params[k]) for k in ['description', 'email', 'name', 'ttl', 'type'] if self.params[k] is not None))
        zones = self.conn.dns.zones(**kwargs)
        self.exit_json(changed=False, zones=[z.to_dict(computed=False) for z in zones])