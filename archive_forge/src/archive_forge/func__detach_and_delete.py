from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _detach_and_delete(self):
    ips = self._find_ips(server=self.server, floating_ip_address=self.params['floating_ip_address'], network_id=self.network['id'] if self.network else None, fixed_address=self.params['fixed_address'], nat_destination_name_or_id=self.params['nat_destination'])
    if not ips:
        self.exit_json(changed=False)
    changed = False
    for ip in ips:
        if ip['fixed_ip_address']:
            self.conn.detach_ip_from_server(server_id=self.server['id'], floating_ip_id=ip['id'])
            changed = True
        if self.params['purge']:
            self.conn.network.delete_ip(ip['id'])
            changed = True
    self.exit_json(changed=changed)