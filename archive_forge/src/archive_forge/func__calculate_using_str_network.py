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
def _calculate_using_str_network(self, ifaces, str_net, security_groups=None):
    add_nets = []
    remove_ports = [iface.port_id for iface in ifaces or []]
    if str_net == self.NETWORK_AUTO:
        nets = self._get_available_networks()
        if not nets:
            nets = [self._auto_allocate_network()]
        if len(nets) > 1:
            msg = 'Multiple possible networks found.'
            raise exception.UnableToAutoAllocateNetwork(message=msg)
        handle_args = {'port_id': None, 'net_id': nets[0], 'fip': None}
        if security_groups:
            sg_ids = self.client_plugin('neutron').get_secgroup_uuids(security_groups)
            handle_args['security_groups'] = sg_ids
        add_nets.append(handle_args)
    return (remove_ports, add_nets)