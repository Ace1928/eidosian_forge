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
def _ipaddress(self):
    if self.ipaddress is None and self.resource_id is not None:
        try:
            ips = self.neutron().show_floatingip(self.resource_id)
        except Exception as ex:
            self.client_plugin('neutron').ignore_not_found(ex)
        else:
            self.ipaddress = ips['floatingip']['floating_ip_address']
    return self.ipaddress or ''