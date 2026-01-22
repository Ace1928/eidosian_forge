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
def _update_networks(self, server, after_props):
    updaters = []
    new_networks = after_props[self.NETWORKS]
    old_networks = self.properties[self.NETWORKS]
    security_groups = after_props[self.SECURITY_GROUPS]
    if not server:
        server = self.client().servers.get(self.resource_id)
    interfaces = server.interface_list()
    remove_ports, add_nets = self.calculate_networks(old_networks, new_networks, interfaces, security_groups)
    for port in remove_ports:
        updaters.append(progress.ServerUpdateProgress(self.resource_id, 'interface_detach', handler_extra={'args': (port,)}, checker_extra={'args': (port,)}))
    for args in add_nets:
        updaters.append(progress.ServerUpdateProgress(self.resource_id, 'interface_attach', handler_extra={'kwargs': args}, checker_extra={'args': (args['port_id'],)}))
    return updaters