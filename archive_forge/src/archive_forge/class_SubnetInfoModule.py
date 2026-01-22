from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class SubnetInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(aliases=['subnet']), filters=dict(type='dict'))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = {}
        subnets = []
        if self.params['name']:
            kwargs['name'] = self.params['name']
            try:
                raw = self.conn.network.get_subnet(self.params['name'])
                raw = raw.to_dict(computed=False)
                subnets.append(raw)
                self.exit(changed=False, subnets=subnets)
            except self.sdk.exceptions.ResourceNotFound:
                pass
        if self.params['filters']:
            kwargs.update(self.params['filters'])
        subnets = self.conn.network.subnets(**kwargs)
        subnets = [i.to_dict(computed=False) for i in subnets]
        self.exit(changed=False, subnets=subnets)