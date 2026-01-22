from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _get_rule_order(self):
    perms = self._get_role_perm()
    rules = []
    if perms:
        for i, rule in enumerate(perms):
            rules.append(rule['id'])
    return rules