from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ResourcesModule(OpenStackModule):
    argument_spec = dict(parameters=dict(type='dict'), service=dict(required=True), type=dict(required=True))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        service_name = self.params['service']
        type_name = self.params['type']
        session = getattr(self.conn, service_name)
        list_function = getattr(session, '{0}s'.format(type_name))
        parameters = self.params['parameters']
        resources = list_function(**parameters) if parameters else list_function()
        self.exit_json(changed=False, resources=[r.to_dict(computed=False) for r in resources])