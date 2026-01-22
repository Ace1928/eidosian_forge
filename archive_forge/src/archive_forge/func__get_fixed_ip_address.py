from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
def _get_fixed_ip_address(self):
    if self.fixed_ip_address is None:
        port = self.client().show_port(self.resource_id)['port']
        if port['fixed_ips'] and len(port['fixed_ips']) > 0:
            self.fixed_ip_address = port['fixed_ips'][0]['ip_address']
    return self.fixed_ip_address