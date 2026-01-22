from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackFirewall(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackFirewall, self).__init__(module)
        self.returns = {'cidrlist': 'cidr', 'startport': 'start_port', 'endport': 'end_port', 'protocol': 'protocol', 'ipaddress': 'ip_address', 'icmpcode': 'icmp_code', 'icmptype': 'icmp_type'}
        self.firewall_rule = None
        self.network = None

    def get_firewall_rule(self):
        if not self.firewall_rule:
            cidrs = self.module.params.get('cidrs')
            protocol = self.module.params.get('protocol')
            start_port = self.module.params.get('start_port')
            end_port = self.get_or_fallback('end_port', 'start_port')
            icmp_code = self.module.params.get('icmp_code')
            icmp_type = self.module.params.get('icmp_type')
            fw_type = self.module.params.get('type')
            if protocol in ['tcp', 'udp'] and (not (start_port and end_port)):
                self.module.fail_json(msg="missing required argument for protocol '%s': start_port or end_port" % protocol)
            if protocol == 'icmp' and (not icmp_type):
                self.module.fail_json(msg="missing required argument for protocol 'icmp': icmp_type")
            if protocol == 'all' and fw_type != 'egress':
                self.module.fail_json(msg="protocol 'all' could only be used for type 'egress'")
            args = {'account': self.get_account('name'), 'domainid': self.get_domain('id'), 'projectid': self.get_project('id'), 'fetch_list': True}
            if fw_type == 'egress':
                args['networkid'] = self.get_network(key='id')
                if not args['networkid']:
                    self.module.fail_json(msg='missing required argument for type egress: network')
                network_cidr = self.get_network(key='cidr')
                egress_cidrs = [network_cidr if cidr == '0.0.0.0/0' else cidr for cidr in cidrs]
                firewall_rules = self.query_api('listEgressFirewallRules', **args)
            else:
                args['ipaddressid'] = self.get_ip_address('id')
                if not args['ipaddressid']:
                    self.module.fail_json(msg='missing required argument for type ingress: ip_address')
                egress_cidrs = None
                firewall_rules = self.query_api('listFirewallRules', **args)
            if firewall_rules:
                for rule in firewall_rules:
                    type_match = self._type_cidrs_match(rule, cidrs, egress_cidrs)
                    protocol_match = self._tcp_udp_match(rule, protocol, start_port, end_port) or self._icmp_match(rule, protocol, icmp_code, icmp_type) or self._egress_all_match(rule, protocol, fw_type)
                    if type_match and protocol_match:
                        self.firewall_rule = rule
                        break
        return self.firewall_rule

    def _tcp_udp_match(self, rule, protocol, start_port, end_port):
        return protocol in ['tcp', 'udp'] and protocol == rule['protocol'] and (start_port == int(rule['startport'])) and (end_port == int(rule['endport']))

    def _egress_all_match(self, rule, protocol, fw_type):
        return protocol in ['all'] and protocol == rule['protocol'] and (fw_type == 'egress')

    def _icmp_match(self, rule, protocol, icmp_code, icmp_type):
        return protocol == 'icmp' and protocol == rule['protocol'] and (icmp_code == rule['icmpcode']) and (icmp_type == rule['icmptype'])

    def _type_cidrs_match(self, rule, cidrs, egress_cidrs):
        if egress_cidrs is not None:
            return ','.join(egress_cidrs) == rule['cidrlist'] or ','.join(cidrs) == rule['cidrlist']
        else:
            return ','.join(cidrs) == rule['cidrlist']

    def create_firewall_rule(self):
        firewall_rule = self.get_firewall_rule()
        if not firewall_rule:
            self.result['changed'] = True
            args = {'cidrlist': self.module.params.get('cidrs'), 'protocol': self.module.params.get('protocol'), 'startport': self.module.params.get('start_port'), 'endport': self.get_or_fallback('end_port', 'start_port'), 'icmptype': self.module.params.get('icmp_type'), 'icmpcode': self.module.params.get('icmp_code')}
            fw_type = self.module.params.get('type')
            if not self.module.check_mode:
                if fw_type == 'egress':
                    args['networkid'] = self.get_network(key='id')
                    res = self.query_api('createEgressFirewallRule', **args)
                else:
                    args['ipaddressid'] = self.get_ip_address('id')
                    res = self.query_api('createFirewallRule', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    firewall_rule = self.poll_job(res, 'firewallrule')
        if firewall_rule:
            firewall_rule = self.ensure_tags(resource=firewall_rule, resource_type='Firewallrule')
            self.firewall_rule = firewall_rule
        return firewall_rule

    def remove_firewall_rule(self):
        firewall_rule = self.get_firewall_rule()
        if firewall_rule:
            self.result['changed'] = True
            args = {'id': firewall_rule['id']}
            fw_type = self.module.params.get('type')
            if not self.module.check_mode:
                if fw_type == 'egress':
                    res = self.query_api('deleteEgressFirewallRule', **args)
                else:
                    res = self.query_api('deleteFirewallRule', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'firewallrule')
        return firewall_rule

    def get_result(self, resource):
        super(AnsibleCloudStackFirewall, self).get_result(resource)
        if resource:
            self.result['type'] = self.module.params.get('type')
            if self.result['type'] == 'egress':
                self.result['network'] = self.get_network(key='displaytext')
            if 'cidrlist' in resource:
                self.result['cidrs'] = resource['cidrlist'].split(',') or [resource['cidrlist']]
        return self.result