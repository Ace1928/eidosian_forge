from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.ec2 import vpc
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
class SubnetRouteTableAssociation(resource.Resource):
    PROPERTIES = ROUTE_TABLE_ID, SUBNET_ID = ('RouteTableId', 'SubnetId')
    properties_schema = {ROUTE_TABLE_ID: properties.Schema(properties.Schema.STRING, _('Route table ID.'), required=True), SUBNET_ID: properties.Schema(properties.Schema.STRING, _('Subnet ID.'), required=True, constraints=[constraints.CustomConstraint('neutron.subnet')])}
    default_client_name = 'neutron'

    def handle_create(self):
        client = self.client()
        subnet_id = self.properties.get(self.SUBNET_ID)
        router_id = self.properties.get(self.ROUTE_TABLE_ID)
        with self.client_plugin().ignore_not_found:
            previous_router = self._router_for_subnet(subnet_id)
            if previous_router:
                client.remove_interface_router(previous_router['id'], {'subnet_id': subnet_id})
        client.add_interface_router(router_id, {'subnet_id': subnet_id})

    def _router_for_subnet(self, subnet_id):
        client = self.client()
        subnet = client.show_subnet(subnet_id)['subnet']
        network_id = subnet['network_id']
        return vpc.VPC.router_for_vpc(client, network_id)

    def handle_delete(self):
        client = self.client()
        subnet_id = self.properties.get(self.SUBNET_ID)
        router_id = self.properties.get(self.ROUTE_TABLE_ID)
        with self.client_plugin().ignore_not_found:
            client.remove_interface_router(router_id, {'subnet_id': subnet_id})
        with self.client_plugin().ignore_not_found:
            default_router = self._router_for_subnet(subnet_id)
            if default_router:
                client.add_interface_router(default_router['id'], {'subnet_id': subnet_id})