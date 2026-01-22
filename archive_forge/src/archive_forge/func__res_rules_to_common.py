from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
def _res_rules_to_common(self, api_rules):
    rules = {}
    for nr in api_rules:
        rule = {}
        rule[self.sg.RULE_FROM_PORT] = nr['port_range_min']
        rule[self.sg.RULE_TO_PORT] = nr['port_range_max']
        rule[self.sg.RULE_IP_PROTOCOL] = nr['protocol']
        rule['direction'] = nr['direction']
        rule[self.sg.RULE_CIDR_IP] = nr['remote_ip_prefix']
        rule[self.sg.RULE_SOURCE_SECURITY_GROUP_ID] = nr['remote_group_id']
        rules[nr['id']] = rule
    return rules