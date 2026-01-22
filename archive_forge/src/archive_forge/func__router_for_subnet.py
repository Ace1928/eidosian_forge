from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.ec2 import vpc
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
def _router_for_subnet(self, subnet_id):
    client = self.client()
    subnet = client.show_subnet(subnet_id)['subnet']
    network_id = subnet['network_id']
    return vpc.VPC.router_for_vpc(client, network_id)