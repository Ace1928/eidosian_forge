from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
class AnsibleHCloudLoadBalancerService(AnsibleHCloud):
    represent = 'hcloud_load_balancer_service'
    hcloud_load_balancer: BoundLoadBalancer | None = None
    hcloud_load_balancer_service: LoadBalancerService | None = None

    def _prepare_result(self):
        http = None
        if self.hcloud_load_balancer_service.protocol != 'tcp':
            http = {'cookie_name': to_native(self.hcloud_load_balancer_service.http.cookie_name), 'cookie_lifetime': self.hcloud_load_balancer_service.http.cookie_name, 'redirect_http': self.hcloud_load_balancer_service.http.redirect_http, 'sticky_sessions': self.hcloud_load_balancer_service.http.sticky_sessions, 'certificates': [to_native(certificate.name) for certificate in self.hcloud_load_balancer_service.http.certificates]}
        health_check = {'protocol': to_native(self.hcloud_load_balancer_service.health_check.protocol), 'port': self.hcloud_load_balancer_service.health_check.port, 'interval': self.hcloud_load_balancer_service.health_check.interval, 'timeout': self.hcloud_load_balancer_service.health_check.timeout, 'retries': self.hcloud_load_balancer_service.health_check.retries}
        if self.hcloud_load_balancer_service.health_check.protocol != 'tcp':
            health_check['http'] = {'domain': to_native(self.hcloud_load_balancer_service.health_check.http.domain), 'path': to_native(self.hcloud_load_balancer_service.health_check.http.path), 'response': to_native(self.hcloud_load_balancer_service.health_check.http.response), 'status_codes': [to_native(status_code) for status_code in self.hcloud_load_balancer_service.health_check.http.status_codes], 'tls': self.hcloud_load_balancer_service.health_check.http.tls}
        return {'load_balancer': to_native(self.hcloud_load_balancer.name), 'protocol': to_native(self.hcloud_load_balancer_service.protocol), 'listen_port': self.hcloud_load_balancer_service.listen_port, 'destination_port': self.hcloud_load_balancer_service.destination_port, 'proxyprotocol': self.hcloud_load_balancer_service.proxyprotocol, 'http': http, 'health_check': health_check}

    def _get_load_balancer(self):
        try:
            self.hcloud_load_balancer = self._client_get_by_name_or_id('load_balancers', self.module.params.get('load_balancer'))
            self._get_load_balancer_service()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _create_load_balancer_service(self):
        self.module.fail_on_missing_params(required_params=['protocol'])
        if self.module.params.get('protocol') == 'tcp':
            self.module.fail_on_missing_params(required_params=['destination_port'])
        params = {'protocol': self.module.params.get('protocol'), 'listen_port': self.module.params.get('listen_port'), 'proxyprotocol': self.module.params.get('proxyprotocol')}
        if self.module.params.get('destination_port'):
            params['destination_port'] = self.module.params.get('destination_port')
        if self.module.params.get('http'):
            params['http'] = self.__get_service_http(http_arg=self.module.params.get('http'))
        if self.module.params.get('health_check'):
            params['health_check'] = self.__get_service_health_checks(health_check=self.module.params.get('health_check'))
        if not self.module.check_mode:
            try:
                self.hcloud_load_balancer.add_service(LoadBalancerService(**params)).wait_until_finished(max_retries=1000)
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_load_balancer()
        self._get_load_balancer_service()

    def __get_service_http(self, http_arg):
        service_http = LoadBalancerServiceHttp(certificates=[])
        if http_arg.get('cookie_name') is not None:
            service_http.cookie_name = http_arg.get('cookie_name')
        if http_arg.get('cookie_lifetime') is not None:
            service_http.cookie_lifetime = http_arg.get('cookie_lifetime')
        if http_arg.get('sticky_sessions') is not None:
            service_http.sticky_sessions = http_arg.get('sticky_sessions')
        if http_arg.get('redirect_http') is not None:
            service_http.redirect_http = http_arg.get('redirect_http')
        if http_arg.get('certificates') is not None:
            certificates = http_arg.get('certificates')
            if certificates is not None:
                for certificate in certificates:
                    hcloud_cert = None
                    try:
                        try:
                            hcloud_cert = self.client.certificates.get_by_name(certificate)
                        except Exception:
                            hcloud_cert = self.client.certificates.get_by_id(certificate)
                    except HCloudException as exception:
                        self.fail_json_hcloud(exception)
                    service_http.certificates.append(hcloud_cert)
        return service_http

    def __get_service_health_checks(self, health_check):
        service_health_check = LoadBalancerHealthCheck()
        if health_check.get('protocol') is not None:
            service_health_check.protocol = health_check.get('protocol')
        if health_check.get('port') is not None:
            service_health_check.port = health_check.get('port')
        if health_check.get('interval') is not None:
            service_health_check.interval = health_check.get('interval')
        if health_check.get('timeout') is not None:
            service_health_check.timeout = health_check.get('timeout')
        if health_check.get('retries') is not None:
            service_health_check.retries = health_check.get('retries')
        if health_check.get('http') is not None:
            health_check_http = health_check.get('http')
            service_health_check.http = LoadBalancerHealtCheckHttp()
            if health_check_http.get('domain') is not None:
                service_health_check.http.domain = health_check_http.get('domain')
            if health_check_http.get('path') is not None:
                service_health_check.http.path = health_check_http.get('path')
            if health_check_http.get('response') is not None:
                service_health_check.http.response = health_check_http.get('response')
            if health_check_http.get('status_codes') is not None:
                service_health_check.http.status_codes = health_check_http.get('status_codes')
            if health_check_http.get('tls') is not None:
                service_health_check.http.tls = health_check_http.get('tls')
        return service_health_check

    def _update_load_balancer_service(self):
        changed = False
        try:
            params = {'listen_port': self.module.params.get('listen_port')}
            if self.module.params.get('destination_port') is not None:
                if self.hcloud_load_balancer_service.destination_port != self.module.params.get('destination_port'):
                    params['destination_port'] = self.module.params.get('destination_port')
                    changed = True
            if self.module.params.get('protocol') is not None:
                if self.hcloud_load_balancer_service.protocol != self.module.params.get('protocol'):
                    params['protocol'] = self.module.params.get('protocol')
                    changed = True
            if self.module.params.get('proxyprotocol') is not None:
                if self.hcloud_load_balancer_service.proxyprotocol != self.module.params.get('proxyprotocol'):
                    params['proxyprotocol'] = self.module.params.get('proxyprotocol')
                    changed = True
            if self.module.params.get('http') is not None:
                params['http'] = self.__get_service_http(http_arg=self.module.params.get('http'))
                changed = True
            if self.module.params.get('health_check') is not None:
                params['health_check'] = self.__get_service_health_checks(health_check=self.module.params.get('health_check'))
                changed = True
            if not self.module.check_mode:
                self.hcloud_load_balancer.update_service(LoadBalancerService(**params)).wait_until_finished(max_retries=1000)
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
        self._get_load_balancer()
        if changed:
            self._mark_as_changed()

    def _get_load_balancer_service(self):
        for service in self.hcloud_load_balancer.services:
            if self.module.params.get('listen_port') == service.listen_port:
                self.hcloud_load_balancer_service = service

    def present_load_balancer_service(self):
        self._get_load_balancer()
        if self.hcloud_load_balancer_service is None:
            self._create_load_balancer_service()
        else:
            self._update_load_balancer_service()

    def delete_load_balancer_service(self):
        try:
            self._get_load_balancer()
            if self.hcloud_load_balancer_service is not None:
                if not self.module.check_mode:
                    try:
                        self.hcloud_load_balancer.delete_service(self.hcloud_load_balancer_service).wait_until_finished(max_retries=1000)
                    except HCloudException as exception:
                        self.fail_json_hcloud(exception)
                self._mark_as_changed()
            self.hcloud_load_balancer_service = None
        except APIException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(load_balancer={'type': 'str', 'required': True}, listen_port={'type': 'int', 'required': True}, destination_port={'type': 'int'}, protocol={'type': 'str', 'choices': ['http', 'https', 'tcp']}, proxyprotocol={'type': 'bool', 'default': False}, http={'type': 'dict', 'options': dict(cookie_name={'type': 'str'}, cookie_lifetime={'type': 'int'}, sticky_sessions={'type': 'bool', 'default': False}, redirect_http={'type': 'bool', 'default': False}, certificates={'type': 'list', 'elements': 'str'})}, health_check={'type': 'dict', 'options': dict(protocol={'type': 'str', 'choices': ['http', 'https', 'tcp']}, port={'type': 'int'}, interval={'type': 'int'}, timeout={'type': 'int'}, retries={'type': 'int'}, http={'type': 'dict', 'options': dict(domain={'type': 'str'}, path={'type': 'str'}, response={'type': 'str'}, status_codes={'type': 'list', 'elements': 'str'}, tls={'type': 'bool', 'default': False})})}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), supports_check_mode=True)