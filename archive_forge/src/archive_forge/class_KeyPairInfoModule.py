from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class KeyPairInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(), user_id=dict(), limit=dict(type='int'), marker=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        filters = {k: self.params[k] for k in ['user_id', 'name', 'limit', 'marker'] if self.params[k] is not None}
        keypairs = self.conn.search_keypairs(name_or_id=self.params['name'], filters=filters)
        result = [raw.to_dict(computed=False) for raw in keypairs]
        self.exit(changed=False, keypairs=result)