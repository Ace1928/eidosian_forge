from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer
class AnsibleHCloudLoadBalancerInfo(AnsibleHCloud):
    represent = 'hcloud_load_balancer_info'
    hcloud_load_balancer_info: list[BoundLoadBalancer] | None = None

    def _prepare_result(self):
        tmp = []
        for load_balancer in self.hcloud_load_balancer_info:
            if load_balancer is not None:
                services = [self._prepare_service_result(service) for service in load_balancer.services]
                targets = [self._prepare_target_result(target) for target in load_balancer.targets]
                private_ipv4_address = None if len(load_balancer.private_net) == 0 else to_native(load_balancer.private_net[0].ip)
                tmp.append({'id': to_native(load_balancer.id), 'name': to_native(load_balancer.name), 'ipv4_address': to_native(load_balancer.public_net.ipv4.ip), 'ipv6_address': to_native(load_balancer.public_net.ipv6.ip), 'private_ipv4_address': private_ipv4_address, 'load_balancer_type': to_native(load_balancer.load_balancer_type.name), 'location': to_native(load_balancer.location.name), 'labels': load_balancer.labels, 'delete_protection': load_balancer.protection['delete'], 'disable_public_interface': False if load_balancer.public_net.enabled else True, 'targets': targets, 'services': services})
        return tmp

    @staticmethod
    def _prepare_service_result(service):
        http = None
        if service.protocol != 'tcp':
            http = {'cookie_name': to_native(service.http.cookie_name), 'cookie_lifetime': service.http.cookie_name, 'redirect_http': service.http.redirect_http, 'sticky_sessions': service.http.sticky_sessions, 'certificates': [to_native(certificate.name) for certificate in service.http.certificates]}
        health_check = {'protocol': to_native(service.health_check.protocol), 'port': service.health_check.port, 'interval': service.health_check.interval, 'timeout': service.health_check.timeout, 'retries': service.health_check.retries}
        if service.health_check.protocol != 'tcp':
            health_check['http'] = {'domain': to_native(service.health_check.http.domain), 'path': to_native(service.health_check.http.path), 'response': to_native(service.health_check.http.response), 'certificates': [to_native(status_code) for status_code in service.health_check.http.status_codes], 'tls': service.health_check.http.tls}
        return {'protocol': to_native(service.protocol), 'listen_port': service.listen_port, 'destination_port': service.destination_port, 'proxyprotocol': service.proxyprotocol, 'http': http, 'health_check': health_check}

    @staticmethod
    def _prepare_target_result(target):
        result = {'type': to_native(target.type), 'use_private_ip': target.use_private_ip}
        if target.type == 'server':
            result['server'] = to_native(target.server.name)
        elif target.type == 'label_selector':
            result['label_selector'] = to_native(target.label_selector.selector)
        elif target.type == 'ip':
            result['ip'] = to_native(target.ip.ip)
        if target.health_status is not None:
            result['health_status'] = [{'listen_port': item.listen_port, 'status': item.status} for item in target.health_status]
        return result

    def get_load_balancers(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_load_balancer_info = [self.client.load_balancers.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_load_balancer_info = [self.client.load_balancers.get_by_name(self.module.params.get('name'))]
            else:
                params = {}
                label_selector = self.module.params.get('label_selector')
                if label_selector:
                    params['label_selector'] = label_selector
                self.hcloud_load_balancer_info = self.client.load_balancers.get_all(**params)
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, label_selector={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)