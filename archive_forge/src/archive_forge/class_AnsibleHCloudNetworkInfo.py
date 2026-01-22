from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork
class AnsibleHCloudNetworkInfo(AnsibleHCloud):
    represent = 'hcloud_network_info'
    hcloud_network_info: list[BoundNetwork] | None = None

    def _prepare_result(self):
        tmp = []
        for network in self.hcloud_network_info:
            if network is not None:
                subnets = []
                for subnet in network.subnets:
                    prepared_subnet = {'type': subnet.type, 'ip_range': subnet.ip_range, 'network_zone': subnet.network_zone, 'gateway': subnet.gateway}
                    subnets.append(prepared_subnet)
                routes = []
                for route in network.routes:
                    prepared_route = {'destination': route.destination, 'gateway': route.gateway}
                    routes.append(prepared_route)
                servers = []
                for server in network.servers:
                    image = None if server.image is None else to_native(server.image.name)
                    ipv4_address = None if server.public_net.ipv4 is None else to_native(server.public_net.ipv4.ip)
                    ipv6 = None if server.public_net.ipv6 is None else to_native(server.public_net.ipv6.ip)
                    prepared_server = {'id': to_native(server.id), 'name': to_native(server.name), 'ipv4_address': ipv4_address, 'ipv6': ipv6, 'image': image, 'server_type': to_native(server.server_type.name), 'datacenter': to_native(server.datacenter.name), 'location': to_native(server.datacenter.location.name), 'rescue_enabled': server.rescue_enabled, 'backup_window': to_native(server.backup_window), 'labels': server.labels, 'status': to_native(server.status)}
                    servers.append(prepared_server)
                tmp.append({'id': to_native(network.id), 'name': to_native(network.name), 'ip_range': to_native(network.ip_range), 'subnetworks': subnets, 'routes': routes, 'expose_routes_to_vswitch': network.expose_routes_to_vswitch, 'servers': servers, 'labels': network.labels, 'delete_protection': network.protection['delete']})
        return tmp

    def get_networks(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_network_info = [self.client.networks.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_network_info = [self.client.networks.get_by_name(self.module.params.get('name'))]
            elif self.module.params.get('label_selector') is not None:
                self.hcloud_network_info = self.client.networks.get_all(label_selector=self.module.params.get('label_selector'))
            else:
                self.hcloud_network_info = self.client.networks.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, label_selector={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)