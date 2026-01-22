from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class NetworkingFloatingIPModule(OpenStackModule):
    argument_spec = dict(fixed_address=dict(), floating_ip_address=dict(), nat_destination=dict(aliases=['fixed_network', 'internal_network']), network=dict(), purge=dict(type='bool', default=False), reuse=dict(type='bool', default=False), server=dict(required=True), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(required_if=[['state', 'absent', ['floating_ip_address']]], required_by={'floating_ip_address': 'network'})

    def run(self):
        self._init()
        if self.params['state'] == 'present':
            self._create_and_attach()
        else:
            self._detach_and_delete()

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

    def _filter_ips(self, server):

        def _flatten(lists):
            return [item for sublist in lists for item in sublist]
        if server['addresses'] is None:
            server = self.conn.compute.get_server(server)
        if not server['addresses']:
            return []
        return [address['addr'] for address in _flatten(server['addresses'].values()) if address['OS-EXT-IPS:type'] == 'floating']

    def _find_ips(self, server, floating_ip_address, network_id, fixed_address, nat_destination_name_or_id):
        if floating_ip_address:
            ip = self.conn.network.find_ip(floating_ip_address)
            return [ip] if ip else []
        elif not fixed_address and nat_destination_name_or_id:
            return self._find_ips_by_nat_destination(server, nat_destination_name_or_id)
        else:
            return self._find_ips_by_network_id_and_fixed_address(server, fixed_address, network_id)

    def _find_ips_by_nat_destination(self, server, nat_destination_name_or_id):
        if not server['addresses']:
            return None
        nat_destination = self.conn.network.find_network(nat_destination_name_or_id, ignore_missing=False)
        fips_with_nat_destination = [addr for addr in server['addresses'].get(nat_destination['name'], []) if addr['OS-EXT-IPS:type'] == 'floating']
        if not fips_with_nat_destination:
            return None
        return [self.conn.network.find_ip(fip['addr'], ignore_missing=False) for fip in fips_with_nat_destination]

    def _find_ips_by_network_id_and_fixed_address(self, server, fixed_address=None, network_id=None):
        ips = [ip for ip in self.conn.network.ips() if ip['floating_ip_address'] in self._filter_ips(server)]
        matching_ips = []
        for ip in ips:
            if network_id and ip['floating_network_id'] != network_id:
                continue
            if not fixed_address:
                matching_ips.append(ip)
            if fixed_address and ip['fixed_ip_address'] == fixed_address:
                matching_ips.append(ip)
        return matching_ips

    def _init(self):
        server_name_or_id = self.params['server']
        server = self.conn.compute.find_server(server_name_or_id, ignore_missing=False)
        self.server = self.conn.compute.get_server(server)
        network_name_or_id = self.params['network']
        if network_name_or_id:
            self.network = self.conn.network.find_network(name_or_id=network_name_or_id, ignore_missing=False)
        else:
            self.network = None