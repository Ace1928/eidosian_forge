from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _get_alert_policy_id_by_name(clc, module, alias, alert_policy_name):
    """
        Returns the alert policy id for the given alert policy name
        :param clc: the clc-sdk instance to use
        :param module: the AnsibleModule object
        :param alias: the clc account alias
        :param alert_policy_name: the name of the alert policy
        :return: alert_policy_id: the alert policy id
        """
    alert_policy_id = None
    policies = clc.v2.API.Call('GET', '/v2/alertPolicies/%s' % alias)
    if not policies:
        return alert_policy_id
    for policy in policies.get('items'):
        if policy.get('name') == alert_policy_name:
            if not alert_policy_id:
                alert_policy_id = policy.get('id')
            else:
                return module.fail_json(msg='multiple alert policies were found with policy name : %s' % alert_policy_name)
    return alert_policy_id