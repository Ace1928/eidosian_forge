from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _find_ips(self, server, floating_ip_address, network_id, fixed_address, nat_destination_name_or_id):
    if floating_ip_address:
        ip = self.conn.network.find_ip(floating_ip_address)
        return [ip] if ip else []
    elif not fixed_address and nat_destination_name_or_id:
        return self._find_ips_by_nat_destination(server, nat_destination_name_or_id)
    else:
        return self._find_ips_by_network_id_and_fixed_address(server, fixed_address, network_id)