from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ComputeFlavorInfoModule(OpenStackModule):
    argument_spec = dict(ephemeral=dict(), limit=dict(type='int'), name=dict(), ram=dict(), vcpus=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        name = self.params['name']
        filters = dict(((k, self.params[k]) for k in ['ephemeral', 'ram', 'vcpus'] if self.params[k] is not None))
        if name:
            flavor = self.conn.compute.find_flavor(name)
            flavors = [flavor] if flavor else []
        else:
            flavors = list(self.conn.compute.flavors())
        if filters:
            flavors = self.conn.range_search(flavors, filters)
        limit = self.params['limit']
        if limit is not None:
            flavors = flavors[:limit]
        self.exit_json(changed=False, flavors=[f.to_dict(computed=False) for f in flavors])