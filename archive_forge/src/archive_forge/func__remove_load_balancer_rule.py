from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _remove_load_balancer_rule(module, oneandone_conn, load_balancer_id, rule_id):
    """
    Removes a rule from a load_balancer.
    """
    try:
        if module.check_mode:
            rule = oneandone_conn.get_load_balancer_rule(load_balancer_id=load_balancer_id, rule_id=rule_id)
            if rule:
                return True
            return False
        load_balancer = oneandone_conn.remove_load_balancer_rule(load_balancer_id=load_balancer_id, rule_id=rule_id)
        return load_balancer
    except Exception as ex:
        module.fail_json(msg=str(ex))