from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ComputeServiceInfoModule(OpenStackModule):
    argument_spec = dict(binary=dict(), host=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = {k: self.params[k] for k in ['binary', 'host'] if self.params[k] is not None}
        compute_services = self.conn.compute.services(**kwargs)
        self.exit_json(changed=False, compute_services=[s.to_dict(computed=False) for s in compute_services])