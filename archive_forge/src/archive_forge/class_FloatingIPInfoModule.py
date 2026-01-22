from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class FloatingIPInfoModule(OpenStackModule):
    argument_spec = dict(description=dict(), fixed_ip_address=dict(), floating_ip_address=dict(), floating_network=dict(), port=dict(), project=dict(aliases=['project_id']), router=dict(), status=dict(choices=['active', 'down']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        query = dict(((k, self.params[k]) for k in ['description', 'fixed_ip_address', 'floating_ip_address'] if self.params[k] is not None))
        for k in ['port', 'router']:
            if self.params[k]:
                k_id = '{0}_id'.format(k)
                find_name = 'find_{0}'.format(k)
                query[k_id] = getattr(self.conn.network, find_name)(name_or_id=self.params[k], ignore_missing=False)['id']
        floating_network_name_or_id = self.params['floating_network']
        if floating_network_name_or_id:
            query['floating_network_id'] = self.conn.network.find_network(name_or_id=floating_network_name_or_id, ignore_missing=False)['id']
        project_name_or_id = self.params['project']
        if project_name_or_id:
            project = self.conn.identity.find_project(project_name_or_id)
            if project:
                query['project_id'] = project['id']
            else:
                query['project_id'] = project_name_or_id
        status = self.params['status']
        if status:
            query['status'] = status.upper()
        self.exit_json(changed=False, floating_ips=[ip.to_dict(computed=False) for ip in self.conn.network.ips(**query)])