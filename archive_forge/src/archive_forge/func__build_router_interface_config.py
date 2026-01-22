from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from collections import defaultdict
def _build_router_interface_config(self, filters):
    external_fixed_ips = None
    internal_ports_missing = []
    internal_ifaces = []
    ext_fixed_ips = None
    if self.params['external_gateway_info']:
        ext_fixed_ips = self.params['external_gateway_info'].get('external_fixed_ips')
    ext_fixed_ips = ext_fixed_ips or self.params['external_fixed_ips']
    if ext_fixed_ips:
        external_fixed_ips = []
        for iface in ext_fixed_ips:
            subnet = self.conn.network.find_subnet(iface['subnet_id'], ignore_missing=False, **filters)
            fip = dict(subnet_id=subnet.id)
            if 'ip_address' in iface:
                fip['ip_address'] = iface['ip_address']
            external_fixed_ips.append(fip)
    if self.params['interfaces']:
        internal_ips = []
        for iface in self.params['interfaces']:
            if isinstance(iface, str):
                subnet = self.conn.network.find_subnet(iface, ignore_missing=False, **filters)
                internal_ifaces.append(dict(subnet_id=subnet.id))
            elif isinstance(iface, dict):
                subnet = self.conn.network.find_subnet(iface['subnet'], ignore_missing=False, **filters)
                if 'net' not in iface:
                    self.fail('Network name missing from interface definition')
                net = self.conn.network.find_network(iface['net'], ignore_missing=False)
                if 'portip' not in iface:
                    internal_ifaces.append(dict(subnet_id=subnet.id))
                elif not iface['portip']:
                    self.fail(msg='put an ip in portip or remove itfrom list to assign default port to router')
                else:
                    portip = iface['portip']
                    port_kwargs = {'network_id': net.id} if net is not None else {}
                    existing_ports = self.conn.network.ports(**port_kwargs)
                    for port in existing_ports:
                        for fip in port['fixed_ips']:
                            if fip['subnet_id'] != subnet.id or fip['ip_address'] != portip:
                                continue
                            internal_ips.append(fip['ip_address'])
                            internal_ifaces.append(dict(port_id=port.id, subnet_id=subnet.id, ip_address=portip))
                    if portip not in internal_ips:
                        internal_ports_missing.append({'network_id': subnet.network_id, 'fixed_ips': [{'ip_address': portip, 'subnet_id': subnet.id}]})
    return {'external_fixed_ips': external_fixed_ips, 'internal_ports_missing': internal_ports_missing, 'internal_ifaces': internal_ifaces}