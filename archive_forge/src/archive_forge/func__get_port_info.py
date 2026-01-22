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
def _get_port_info(self, ni_id=None, instance_id=None):
    port_id = None
    port_rsrc = None
    if ni_id:
        port_rsrc = self.neutron().list_ports(id=ni_id)['ports'][0]
        port_id = ni_id
    elif instance_id:
        ports = self.neutron().list_ports(device_id=instance_id)
        port_rsrc = ports['ports'][0]
        port_id = port_rsrc['id']
    return (port_id, port_rsrc)