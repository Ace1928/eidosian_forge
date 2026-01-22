from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ServerInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(aliases=['server']), detailed=dict(type='bool', default=False), filters=dict(type='dict'), all_projects=dict(type='bool', default=False))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = dict(((k, self.params[k]) for k in ['detailed', 'filters', 'all_projects'] if self.params[k] is not None))
        kwargs['name_or_id'] = self.params['name']
        self.exit(changed=False, servers=[server.to_dict(computed=False) for server in self.conn.search_servers(**kwargs)])