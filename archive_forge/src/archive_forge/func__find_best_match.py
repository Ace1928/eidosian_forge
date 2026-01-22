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
def _find_best_match(self, existing_interfaces, specified_net):
    specified_net_items = set(specified_net.items())
    if specified_net.get(self.NETWORK_PORT) is not None:
        for iface in existing_interfaces:
            if iface[self.NETWORK_PORT] == specified_net[self.NETWORK_PORT] and specified_net_items.issubset(set(iface.items())):
                return iface
    elif specified_net.get(self.NETWORK_FIXED_IP) is not None:
        for iface in existing_interfaces:
            if iface[self.NETWORK_FIXED_IP] == specified_net[self.NETWORK_FIXED_IP] and specified_net_items.issubset(set(iface.items())):
                return iface
    else:
        best, matches, num = (None, 0, 0)
        for iface in existing_interfaces:
            iface_items = set(iface.items())
            if specified_net_items.issubset(iface_items):
                num = len(specified_net_items.intersection(iface_items))
            if num > matches:
                best, matches = (iface, num)
        return best