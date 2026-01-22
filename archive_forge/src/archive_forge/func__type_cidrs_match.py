from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _type_cidrs_match(self, rule, cidrs, egress_cidrs):
    if egress_cidrs is not None:
        return ','.join(egress_cidrs) == rule['cidrlist'] or ','.join(cidrs) == rule['cidrlist']
    else:
        return ','.join(cidrs) == rule['cidrlist']