import itertools
import eventlet
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from heat.engine.resources.openstack.neutron import port as neutron_port
def update_networks_matching_iface_port(self, old_nets, interfaces):

    def get_iface_props(iface):
        ipaddr = None
        subnet = None
        if len(iface.fixed_ips) > 0:
            ipaddr = iface.fixed_ips[0]['ip_address']
            subnet = iface.fixed_ips[0]['subnet_id']
        return {self.NETWORK_PORT: iface.port_id, self.NETWORK_ID: iface.net_id, self.NETWORK_FIXED_IP: ipaddr, self.NETWORK_SUBNET: subnet}
    interfaces_net_props = [get_iface_props(iface) for iface in interfaces]
    for old_net in old_nets:
        if old_net[self.NETWORK_PORT] is None:
            old_net[self.NETWORK_ID] = self._get_network_id(old_net)
        old_net_reduced = {k: v for k, v in old_net.items() if k in self._IFACE_MANAGED_KEYS and v is not None}
        match = self._find_best_match(interfaces_net_props, old_net_reduced)
        if match is not None:
            old_net.update(match)
            interfaces_net_props.remove(match)