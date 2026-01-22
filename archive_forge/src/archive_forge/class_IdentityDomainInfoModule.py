from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityDomainInfoModule(OpenStackModule):
    argument_spec = dict(filters=dict(type='dict'), name=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = {}
        name = self.params['name']
        if name is not None:
            kwargs['name_or_id'] = name
        filters = self.params['filters']
        if filters is not None:
            kwargs['filters'] = filters
        self.exit_json(changed=False, domains=[d.to_dict(computed=False) for d in self.conn.search_domains(**kwargs)])