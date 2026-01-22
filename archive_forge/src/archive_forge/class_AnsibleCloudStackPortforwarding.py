from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackPortforwarding(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackPortforwarding, self).__init__(module)
        self.returns = {'virtualmachinedisplayname': 'vm_display_name', 'virtualmachinename': 'vm_name', 'ipaddress': 'ip_address', 'vmguestip': 'vm_guest_ip', 'publicip': 'public_ip', 'protocol': 'protocol'}
        self.returns_to_int = {'publicport': 'public_port', 'publicendport': 'public_end_port', 'privateport': 'private_port', 'privateendport': 'private_end_port'}
        self.portforwarding_rule = None

    def get_portforwarding_rule(self):
        if not self.portforwarding_rule:
            protocol = self.module.params.get('protocol')
            public_port = self.module.params.get('public_port')
            args = {'ipaddressid': self.get_ip_address(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
            portforwarding_rules = self.query_api('listPortForwardingRules', **args)
            if portforwarding_rules and 'portforwardingrule' in portforwarding_rules:
                for rule in portforwarding_rules['portforwardingrule']:
                    if protocol == rule['protocol'] and public_port == int(rule['publicport']):
                        self.portforwarding_rule = rule
                        break
        return self.portforwarding_rule

    def present_portforwarding_rule(self):
        portforwarding_rule = self.get_portforwarding_rule()
        if portforwarding_rule:
            portforwarding_rule = self.update_portforwarding_rule(portforwarding_rule)
        else:
            portforwarding_rule = self.create_portforwarding_rule()
        if portforwarding_rule:
            portforwarding_rule = self.ensure_tags(resource=portforwarding_rule, resource_type='PortForwardingRule')
            self.portforwarding_rule = portforwarding_rule
        return portforwarding_rule

    def create_portforwarding_rule(self):
        args = {'protocol': self.module.params.get('protocol'), 'publicport': self.module.params.get('public_port'), 'publicendport': self.get_or_fallback('public_end_port', 'public_port'), 'privateport': self.module.params.get('private_port'), 'privateendport': self.get_or_fallback('private_end_port', 'private_port'), 'openfirewall': self.module.params.get('open_firewall'), 'vmguestip': self.get_vm_guest_ip(), 'ipaddressid': self.get_ip_address(key='id'), 'virtualmachineid': self.get_vm(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'networkid': self.get_network(key='id')}
        portforwarding_rule = None
        self.result['changed'] = True
        if not self.module.check_mode:
            portforwarding_rule = self.query_api('createPortForwardingRule', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                portforwarding_rule = self.poll_job(portforwarding_rule, 'portforwardingrule')
        return portforwarding_rule

    def update_portforwarding_rule(self, portforwarding_rule):
        args = {'protocol': self.module.params.get('protocol'), 'publicport': self.module.params.get('public_port'), 'publicendport': self.get_or_fallback('public_end_port', 'public_port'), 'privateport': self.module.params.get('private_port'), 'privateendport': self.get_or_fallback('private_end_port', 'private_port'), 'vmguestip': self.get_vm_guest_ip(), 'ipaddressid': self.get_ip_address(key='id'), 'virtualmachineid': self.get_vm(key='id'), 'networkid': self.get_network(key='id')}
        if self.has_changed(args, portforwarding_rule):
            self.result['changed'] = True
            if not self.module.check_mode:
                self.absent_portforwarding_rule()
                portforwarding_rule = self.query_api('createPortForwardingRule', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    portforwarding_rule = self.poll_job(portforwarding_rule, 'portforwardingrule')
        return portforwarding_rule

    def absent_portforwarding_rule(self):
        portforwarding_rule = self.get_portforwarding_rule()
        if portforwarding_rule:
            self.result['changed'] = True
            args = {'id': portforwarding_rule['id']}
            if not self.module.check_mode:
                res = self.query_api('deletePortForwardingRule', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'portforwardingrule')
        return portforwarding_rule

    def get_result(self, resource):
        super(AnsibleCloudStackPortforwarding, self).get_result(resource)
        if resource:
            for search_key, return_key in self.returns_to_int.items():
                if search_key in resource:
                    self.result[return_key] = int(resource[search_key])
        return self.result