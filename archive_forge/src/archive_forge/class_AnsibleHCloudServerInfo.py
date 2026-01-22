from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.servers import BoundServer
class AnsibleHCloudServerInfo(AnsibleHCloud):
    represent = 'hcloud_server_info'
    hcloud_server_info: list[BoundServer] | None = None

    def _prepare_result(self):
        tmp = []
        for server in self.hcloud_server_info:
            if server is not None:
                image = None if server.image is None else to_native(server.image.name)
                placement_group = None if server.placement_group is None else to_native(server.placement_group.name)
                ipv4_address = None if server.public_net.ipv4 is None else to_native(server.public_net.ipv4.ip)
                ipv6 = None if server.public_net.ipv6 is None else to_native(server.public_net.ipv6.ip)
                backup_window = None if server.backup_window is None else to_native(server.backup_window)
                tmp.append({'id': to_native(server.id), 'name': to_native(server.name), 'created': to_native(server.created.isoformat()), 'ipv4_address': ipv4_address, 'ipv6': ipv6, 'private_networks': [to_native(net.network.name) for net in server.private_net], 'private_networks_info': [{'name': to_native(net.network.name), 'ip': net.ip} for net in server.private_net], 'image': image, 'server_type': to_native(server.server_type.name), 'datacenter': to_native(server.datacenter.name), 'location': to_native(server.datacenter.location.name), 'placement_group': placement_group, 'rescue_enabled': server.rescue_enabled, 'backup_window': backup_window, 'labels': server.labels, 'status': to_native(server.status), 'delete_protection': server.protection['delete'], 'rebuild_protection': server.protection['rebuild']})
        return tmp

    def get_servers(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_server_info = [self.client.servers.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_server_info = [self.client.servers.get_by_name(self.module.params.get('name'))]
            elif self.module.params.get('label_selector') is not None:
                self.hcloud_server_info = self.client.servers.get_all(label_selector=self.module.params.get('label_selector'))
            else:
                self.hcloud_server_info = self.client.servers.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, label_selector={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)