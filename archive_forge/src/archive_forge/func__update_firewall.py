from __future__ import annotations
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
def _update_firewall(self):
    name = self.module.params.get('name')
    if name is not None and self.hcloud_firewall.name != name:
        self.module.fail_on_missing_params(required_params=['id'])
        if not self.module.check_mode:
            self.hcloud_firewall.update(name=name)
        self._mark_as_changed()
    labels = self.module.params.get('labels')
    if labels is not None and self.hcloud_firewall.labels != labels:
        if not self.module.check_mode:
            self.hcloud_firewall.update(labels=labels)
        self._mark_as_changed()
    rules = self.module.params.get('rules')
    if rules is not None and rules != [self._prepare_result_rule(rule) for rule in self.hcloud_firewall.rules]:
        if not self.module.check_mode:
            new_rules = [FirewallRule(direction=rule['direction'], protocol=rule['protocol'], source_ips=rule['source_ips'] if rule['source_ips'] is not None else [], destination_ips=rule['destination_ips'] if rule['destination_ips'] is not None else [], port=rule['port'], description=rule['description']) for rule in rules]
            self.hcloud_firewall.set_rules(new_rules)
        self._mark_as_changed()
    self._get_firewall()