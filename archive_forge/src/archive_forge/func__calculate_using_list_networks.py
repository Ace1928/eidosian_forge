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
def _calculate_using_list_networks(self, old_nets, new_nets, ifaces, security_groups):
    remove_ports = []
    add_nets = []
    if not new_nets and (not old_nets):
        return (remove_ports, add_nets)
    new_nets = new_nets or []
    old_nets = old_nets or []
    remove_ports, not_updated_nets = self._calculate_remove_ports(old_nets, new_nets, ifaces)
    add_nets = self._calculate_add_nets(new_nets, not_updated_nets, security_groups)
    return (remove_ports, add_nets)