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
def _update_network_interfaces(self, server, prop_diff):
    updaters = []
    new_network_ifaces = prop_diff.get(self.NETWORK_INTERFACES)
    old_network_ifaces = self.properties.get(self.NETWORK_INTERFACES)
    if old_network_ifaces and new_network_ifaces:
        self._remove_matched_ifaces(old_network_ifaces, new_network_ifaces)
        if old_network_ifaces:
            old_nics = self._build_nics(old_network_ifaces)
            for nic in old_nics:
                updaters.append(progress.ServerUpdateProgress(self.resource_id, 'interface_detach', complete=True, handler_extra={'args': (nic['port-id'],)}))
        if new_network_ifaces:
            new_nics = self._build_nics(new_network_ifaces)
            for nic in new_nics:
                handler_kwargs = {'port_id': nic['port-id']}
                updaters.append(progress.ServerUpdateProgress(self.resource_id, 'interface_attach', complete=True, handler_extra={'kwargs': handler_kwargs}))
    elif old_network_ifaces and self.NETWORK_INTERFACES not in prop_diff:
        LOG.warning('There is no change of "%(net_interfaces)s" for instance %(server)s, do nothing when updating.', {'net_interfaces': self.NETWORK_INTERFACES, 'server': self.resource_id})
    else:
        subnet_id = prop_diff.get(self.SUBNET_ID) or self.properties.get(self.SUBNET_ID)
        security_groups = self._get_security_groups()
        if not server:
            server = self.client().servers.get(self.resource_id)
        interfaces = server.interface_list()
        for iface in interfaces:
            updaters.append(progress.ServerUpdateProgress(self.resource_id, 'interface_detach', complete=True, handler_extra={'args': (iface.port_id,)}))
        self._port_data_delete()
        nics = self._build_nics(new_network_ifaces, security_groups=security_groups, subnet_id=subnet_id)
        if not nics:
            updaters.append(progress.ServerUpdateProgress(self.resource_id, 'interface_attach', complete=True))
        else:
            for nic in nics:
                updaters.append(progress.ServerUpdateProgress(self.resource_id, 'interface_attach', complete=True, handler_extra={'kwargs': {'port_id': nic['port-id']}}))
    return updaters