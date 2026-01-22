from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class CatalogServiceInfoModule(OpenStackModule):
    argument_spec = dict(name=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        name_or_id = self.params['name']
        if name_or_id:
            service = self.conn.identity.find_service(name_or_id)
            services = [service] if service else []
        else:
            services = self.conn.identity.services()
        self.exit_json(changed=False, services=[s.to_dict(computed=False) for s in services])