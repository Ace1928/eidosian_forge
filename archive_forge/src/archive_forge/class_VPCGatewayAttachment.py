from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.ec2 import route_table
class VPCGatewayAttachment(resource.Resource):
    PROPERTIES = VPC_ID, INTERNET_GATEWAY_ID, VPN_GATEWAY_ID = ('VpcId', 'InternetGatewayId', 'VpnGatewayId')
    properties_schema = {VPC_ID: properties.Schema(properties.Schema.STRING, _('VPC ID for this gateway association.'), required=True), INTERNET_GATEWAY_ID: properties.Schema(properties.Schema.STRING, _('ID of the InternetGateway.')), VPN_GATEWAY_ID: properties.Schema(properties.Schema.STRING, _('ID of the VPNGateway to attach to the VPC.'), implemented=False)}
    default_client_name = 'neutron'

    def _vpc_route_tables(self, ignore_errors=False):
        for res in self.stack.values():
            if res.has_interface('AWS::EC2::RouteTable'):
                try:
                    vpc_id = self.properties[self.VPC_ID]
                    rt_vpc_id = res.properties.get(route_table.RouteTable.VPC_ID)
                except (ValueError, TypeError):
                    if ignore_errors:
                        continue
                    else:
                        raise
                if rt_vpc_id == vpc_id:
                    yield res

    def add_dependencies(self, deps):
        super(VPCGatewayAttachment, self).add_dependencies(deps)
        for route_tbl in self._vpc_route_tables(ignore_errors=True):
            deps += (self, route_tbl)

    def handle_create(self):
        client = self.client()
        external_network_id = InternetGateway.get_external_network_id(client)
        for router in self._vpc_route_tables():
            client.add_gateway_router(router.resource_id, {'network_id': external_network_id})

    def handle_delete(self):
        for router in self._vpc_route_tables():
            with self.client_plugin().ignore_not_found:
                self.client().remove_gateway_router(router.resource_id)