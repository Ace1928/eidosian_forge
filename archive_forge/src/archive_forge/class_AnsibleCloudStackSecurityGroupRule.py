from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackSecurityGroupRule(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackSecurityGroupRule, self).__init__(module)
        self.returns = {'icmptype': 'icmp_type', 'icmpcode': 'icmp_code', 'endport': 'end_port', 'startport': 'start_port', 'protocol': 'protocol', 'cidr': 'cidr', 'securitygroupname': 'user_security_group'}

    def _tcp_udp_match(self, rule, protocol, start_port, end_port):
        return protocol in ['tcp', 'udp'] and protocol == rule['protocol'] and (start_port == int(rule['startport'])) and (end_port == int(rule['endport']))

    def _icmp_match(self, rule, protocol, icmp_code, icmp_type):
        return protocol == 'icmp' and protocol == rule['protocol'] and (icmp_code == int(rule['icmpcode'])) and (icmp_type == int(rule['icmptype']))

    def _ah_esp_gre_match(self, rule, protocol):
        return protocol in ['ah', 'esp', 'gre'] and protocol == rule['protocol']

    def _type_security_group_match(self, rule, security_group_name):
        return security_group_name and 'securitygroupname' in rule and (security_group_name == rule['securitygroupname'])

    def _type_cidr_match(self, rule, cidr):
        return 'cidr' in rule and cidr == rule['cidr']

    def _get_rule(self, rules):
        user_security_group_name = self.module.params.get('user_security_group')
        cidr = self.module.params.get('cidr')
        protocol = self.module.params.get('protocol')
        start_port = self.module.params.get('start_port')
        end_port = self.get_or_fallback('end_port', 'start_port')
        icmp_code = self.module.params.get('icmp_code')
        icmp_type = self.module.params.get('icmp_type')
        if protocol in ['tcp', 'udp'] and (start_port is None or end_port is None):
            self.module.fail_json(msg="no start_port or end_port set for protocol '%s'" % protocol)
        if protocol == 'icmp' and (icmp_type is None or icmp_code is None):
            self.module.fail_json(msg="no icmp_type or icmp_code set for protocol '%s'" % protocol)
        for rule in rules:
            if user_security_group_name:
                type_match = self._type_security_group_match(rule, user_security_group_name)
            else:
                type_match = self._type_cidr_match(rule, cidr)
            protocol_match = self._tcp_udp_match(rule, protocol, start_port, end_port) or self._icmp_match(rule, protocol, icmp_code, icmp_type) or self._ah_esp_gre_match(rule, protocol)
            if type_match and protocol_match:
                return rule
        return None

    def get_security_group(self, security_group_name=None):
        if not security_group_name:
            security_group_name = self.module.params.get('security_group')
        args = {'securitygroupname': security_group_name, 'projectid': self.get_project('id')}
        sgs = self.query_api('listSecurityGroups', **args)
        if not sgs or 'securitygroup' not in sgs:
            self.module.fail_json(msg="security group '%s' not found" % security_group_name)
        return sgs['securitygroup'][0]

    def add_rule(self):
        security_group = self.get_security_group()
        args = {}
        user_security_group_name = self.module.params.get('user_security_group')
        if user_security_group_name:
            args['usersecuritygrouplist'] = []
            user_security_group = self.get_security_group(user_security_group_name)
            args['usersecuritygrouplist'].append({'group': user_security_group['name'], 'account': user_security_group['account']})
        else:
            args['cidrlist'] = self.module.params.get('cidr')
        args['protocol'] = self.module.params.get('protocol')
        args['startport'] = self.module.params.get('start_port')
        args['endport'] = self.get_or_fallback('end_port', 'start_port')
        args['icmptype'] = self.module.params.get('icmp_type')
        args['icmpcode'] = self.module.params.get('icmp_code')
        args['projectid'] = self.get_project('id')
        args['securitygroupid'] = security_group['id']
        rule = None
        res = None
        sg_type = self.module.params.get('type')
        if sg_type == 'ingress':
            if 'ingressrule' in security_group:
                rule = self._get_rule(security_group['ingressrule'])
            if not rule:
                self.result['changed'] = True
                if not self.module.check_mode:
                    res = self.query_api('authorizeSecurityGroupIngress', **args)
        elif sg_type == 'egress':
            if 'egressrule' in security_group:
                rule = self._get_rule(security_group['egressrule'])
            if not rule:
                self.result['changed'] = True
                if not self.module.check_mode:
                    res = self.query_api('authorizeSecurityGroupEgress', **args)
        poll_async = self.module.params.get('poll_async')
        if res and poll_async:
            security_group = self.poll_job(res, 'securitygroup')
            key = sg_type + 'rule'
            if key in security_group:
                rule = security_group[key][0]
        return rule

    def remove_rule(self):
        security_group = self.get_security_group()
        rule = None
        res = None
        sg_type = self.module.params.get('type')
        if sg_type == 'ingress':
            rule = self._get_rule(security_group['ingressrule'])
            if rule:
                self.result['changed'] = True
                if not self.module.check_mode:
                    res = self.query_api('revokeSecurityGroupIngress', id=rule['ruleid'])
        elif sg_type == 'egress':
            rule = self._get_rule(security_group['egressrule'])
            if rule:
                self.result['changed'] = True
                if not self.module.check_mode:
                    res = self.query_api('revokeSecurityGroupEgress', id=rule['ruleid'])
        poll_async = self.module.params.get('poll_async')
        if res and poll_async:
            res = self.poll_job(res, 'securitygroup')
        return rule

    def get_result(self, resource):
        super(AnsibleCloudStackSecurityGroupRule, self).get_result(resource)
        self.result['type'] = self.module.params.get('type')
        self.result['security_group'] = self.module.params.get('security_group')
        return self.result