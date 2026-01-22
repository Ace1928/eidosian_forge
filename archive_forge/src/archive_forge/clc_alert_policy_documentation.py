from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib

        retrieves the alert policy id of the account based on the name of the policy
        :param module: the AnsibleModule object
        :param alert_policy_name: the alert policy name
        :return: alert_policy_id: The alert policy id
        