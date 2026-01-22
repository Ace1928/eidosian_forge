from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _add_load_balancer_rules(module, oneandone_conn, load_balancer_id, rules):
    """
    Adds new rules to a load_balancer.
    """
    try:
        load_balancer_rules = []
        for rule in rules:
            load_balancer_rule = oneandone.client.LoadBalancerRule(protocol=rule['protocol'], port_balancer=rule['port_balancer'], port_server=rule['port_server'], source=rule['source'])
            load_balancer_rules.append(load_balancer_rule)
        if module.check_mode:
            lb_id = get_load_balancer(oneandone_conn, load_balancer_id)
            if load_balancer_rules and lb_id:
                return True
            return False
        load_balancer = oneandone_conn.add_load_balancer_rule(load_balancer_id=load_balancer_id, load_balancer_rules=load_balancer_rules)
        return load_balancer
    except Exception as ex:
        module.fail_json(msg=str(ex))