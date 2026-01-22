from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_aa_policy_id(clc, module):
    """
        Validate if the anti affinity policy exist for the given name and throw error if not
        :param clc: the clc-sdk instance
        :param module: the module to validate
        :return: aa_policy_id: the anti affinity policy id of the given name.
        """
    aa_policy_id = module.params.get('anti_affinity_policy_id')
    aa_policy_name = module.params.get('anti_affinity_policy_name')
    if not aa_policy_id and aa_policy_name:
        alias = module.params.get('alias')
        aa_policy_id = ClcServer._get_anti_affinity_policy_id(clc, module, alias, aa_policy_name)
        if not aa_policy_id:
            module.fail_json(msg='No anti affinity policy was found with policy name : %s' % aa_policy_name)
    return aa_policy_id