from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import client_exception
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.ec2 import internet_gateway
from heat.engine.resources.aws.ec2 import vpc
from heat.engine import support
def _handle_update_portInfo(self, prop_diff):
    instance_id_update = prop_diff.get(self.INSTANCE_ID)
    ni_id_update = prop_diff.get(self.NETWORK_INTERFACE_ID)
    eip = self.properties[self.EIP]
    allocation_id = self.properties[self.ALLOCATION_ID]
    if eip:
        self.client_plugin().associate_floatingip_address(instance_id_update, eip)
    else:
        port_id, port_rsrc = self._get_port_info(ni_id_update, instance_id_update)
        if not port_id or not port_rsrc:
            LOG.error('Port not specified.')
            raise exception.NotFound(_('Failed to update, can not found port info.'))
        network_id = port_rsrc['network_id']
        self._neutron_add_gateway_router(allocation_id, network_id)
        self._neutron_update_floating_ip(allocation_id, port_id)