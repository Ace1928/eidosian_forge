import copy
import ipaddress
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import port as neutron_port
from heat.engine.resources.openstack.neutron import subnet
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import server_base
from heat.engine import support
from heat.engine import translation
from heat.rpc import api as rpc_api
def _get_server_addresses(self, server, extend_networks=True):
    """Adds port id, subnets and network attributes to addresses list.

        This method is used only for resolving attributes.
        :param server: The server resource
        :param extend_networks: When False the network is not extended, i.e
                                the net is returned without replacing name on
                                id.
        """
    nets = {}
    ifaces = self.client('neutron').list_ports(device_id=server.id)
    for port in ifaces['ports']:
        net_label = self.client('neutron').list_networks(id=port['network_id'])['networks'][0]['name']
        net = nets.setdefault(net_label, [])
        for fixed_ip in port['fixed_ips']:
            addr = {'addr': fixed_ip.get('ip_address'), 'OS-EXT-IPS-MAC:mac_addr': port['mac_address'], 'OS-EXT-IPS:type': 'fixed', 'port': port['id']}
            try:
                addr['version'] = (ipaddress.ip_address(addr['addr']).version,)
            except ValueError:
                addr['version'] = None
            if addr['addr']:
                fips = self.client('neutron').list_floatingips(fixed_ip_address=addr['addr'])
                for fip in fips['floatingips']:
                    net.append({'addr': fip['floating_ip_address'], 'version': addr['version'], 'OS-EXT-IPS-MAC:mac_addr': port['mac_address'], 'OS-EXT-IPS:type': 'floating', 'port': None})
            if not extend_networks:
                net.append(addr)
                continue
            addr['subnets'] = self._get_subnets_attr(port['fixed_ips'])
            addr['network'] = self._get_network_attr(port['network_id'])
            net.append(addr)
    if extend_networks:
        return self._extend_networks(nets)
    else:
        return nets