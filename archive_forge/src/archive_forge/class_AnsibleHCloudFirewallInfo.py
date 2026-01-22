from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
class AnsibleHCloudFirewallInfo(AnsibleHCloud):
    represent = 'hcloud_firewall_info'
    hcloud_firewall_info: list[BoundFirewall] | None = None

    def _prepare_result(self):
        tmp = []
        for firewall in self.hcloud_firewall_info:
            if firewall is None:
                continue
            tmp.append({'id': to_native(firewall.id), 'name': to_native(firewall.name), 'labels': firewall.labels, 'rules': [self._prepare_result_rule(rule) for rule in firewall.rules], 'applied_to': [self._prepare_result_applied_to(resource) for resource in firewall.applied_to]})
        return tmp

    def _prepare_result_rule(self, rule: FirewallRule):
        return {'description': to_native(rule.description) if rule.description is not None else None, 'direction': to_native(rule.direction), 'protocol': to_native(rule.protocol), 'port': to_native(rule.port) if rule.port is not None else None, 'source_ips': [to_native(cidr) for cidr in rule.source_ips], 'destination_ips': [to_native(cidr) for cidr in rule.destination_ips]}

    def _prepare_result_applied_to(self, resource: FirewallResource):
        result = {'type': to_native(resource.type), 'server': to_native(resource.server.id) if resource.server is not None else None, 'label_selector': to_native(resource.label_selector.selector) if resource.label_selector is not None else None}
        if resource.applied_to_resources is not None:
            result['applied_to_resources'] = [{'type': to_native(item.type), 'server': to_native(item.server.id) if item.server is not None else None} for item in resource.applied_to_resources]
        return result

    def get_firewalls(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_firewall_info = [self.client.firewalls.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_firewall_info = [self.client.firewalls.get_by_name(self.module.params.get('name'))]
            elif self.module.params.get('label_selector') is not None:
                self.hcloud_firewall_info = self.client.firewalls.get_all(label_selector=self.module.params.get('label_selector'))
            else:
                self.hcloud_firewall_info = self.client.firewalls.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, label_selector={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)