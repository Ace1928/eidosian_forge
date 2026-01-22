from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityFederationMappingInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(aliases=['id']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        id = self.params['name']
        if id:
            mapping = self.conn.identity.find_mapping(name_or_id=id, ignore_missing=True)
            mappings = [mapping] if mapping else []
        else:
            mappings = self.conn.identity.mappings()
        self.exit_json(changed=False, mappings=[mapping.to_dict(computed=False) for mapping in mappings])