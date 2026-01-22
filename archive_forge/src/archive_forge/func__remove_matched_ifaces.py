import copy
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import scheduler_hints as sh
def _remove_matched_ifaces(self, old_network_ifaces, new_network_ifaces):
    old_network_ifaces_copy = copy.deepcopy(old_network_ifaces)
    for iface in old_network_ifaces_copy:
        if iface in new_network_ifaces:
            new_network_ifaces.remove(iface)
            old_network_ifaces.remove(iface)