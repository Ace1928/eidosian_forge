from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _type_security_group_match(self, rule, security_group_name):
    return security_group_name and 'securitygroupname' in rule and (security_group_name == rule['securitygroupname'])