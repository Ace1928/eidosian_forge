from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _create_and_attach(self):
    changed = False
    fixed_address = self.params['fixed_address']
    floating_ip_address = self.params['floating_ip_address']
    nat_destination_name_or_id = self.params['nat_destination']
    network_id = self.network['id'] if self.network else None
    ips = self._find_ips(server=self.server, floating_ip_address=floating_ip_address, network_id=network_id, fixed_address=fixed_address, nat_destination_name_or_id=nat_destination_name_or_id)
    ip = ips[0] if ips else None
    if floating_ip_address:
        if not ip:
            self.conn.network.create_ip(floating_ip_address=floating_ip_address, floating_network_id=network_id)
            changed = True
        elif ip.port_details and ip.port_details['status'] == 'ACTIVE' and (floating_ip_address not in self._filter_ips(self.server)):
            self.fail_json(msg='Floating ip {0} has been attached to different server'.format(floating_ip_address))
        if not ip or floating_ip_address not in self._filter_ips(self.server):
            self.conn.add_ip_list(server=self.server, ips=[floating_ip_address], wait=self.params['wait'], timeout=self.params['timeout'], fixed_address=fixed_address)
            changed = True
        else:
            pass
    elif not ips:
        self.conn.add_ips_to_server(server=self.server, ip_pool=network_id, ips=None, reuse=self.params['reuse'], fixed_address=fixed_address, wait=self.params['wait'], timeout=self.params['timeout'], nat_destination=nat_destination_name_or_id)
        changed = True
    else:
        pass
    if changed:
        self.server = self.conn.compute.get_server(self.server)
        ips = self._find_ips(self.server, floating_ip_address, network_id, fixed_address, nat_destination_name_or_id)
    self.exit_json(changed=changed, floating_ip=ips[0].to_dict(computed=False) if ips else None)