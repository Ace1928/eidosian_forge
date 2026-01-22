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
def _handle_update_eipInfo(self, prop_diff):
    eip_update = prop_diff.get(self.EIP)
    allocation_id_update = prop_diff.get(self.ALLOCATION_ID)
    instance_id = self.properties[self.INSTANCE_ID]
    ni_id = self.properties[self.NETWORK_INTERFACE_ID]
    if eip_update:
        self._floatingIp_detach()
        self.client_plugin().associate_floatingip_address(instance_id, eip_update)
        self.resource_id_set(eip_update)
    elif allocation_id_update:
        self._floatingIp_detach()
        port_id, port_rsrc = self._get_port_info(ni_id, instance_id)
        if not port_id or not port_rsrc:
            LOG.error('Port not specified.')
            raise exception.NotFound(_('Failed to update, can not found port info.'))
        network_id = port_rsrc['network_id']
        self._neutron_add_gateway_router(allocation_id_update, network_id)
        self._neutron_update_floating_ip(allocation_id_update, port_id)
        self.resource_id_set(allocation_id_update)