from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _get_alert_policies(self, alias):
    """
        Get the alert policies for account alias by calling the CLC API.
        :param alias: the account alias
        :return: the alert policies for the account alias
        """
    response = {}
    policies = self.clc.v2.API.Call('GET', '/v2/alertPolicies/%s' % alias)
    for policy in policies.get('items'):
        response[policy.get('id')] = policy
    return response