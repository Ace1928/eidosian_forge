from __future__ import absolute_import, division, print_function
import collections
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.validation import check_required_if
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def validate_param_values(module, obj, param=None):
    if not param:
        param = module.params
    for key in obj:
        validator = globals().get('validate_%s' % key)
        if callable(validator):
            validator(param.get(key), module)