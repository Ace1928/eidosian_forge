from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class NetworkInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(), filters=dict(type='dict'))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = {'filters': self.params['filters'], 'name_or_id': self.params['name']}
        networks = self.conn.search_networks(**kwargs)
        networks = [i.to_dict(computed=False) for i in networks]
        self.exit(changed=False, networks=networks)