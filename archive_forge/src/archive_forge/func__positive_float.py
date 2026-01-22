from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.validation import check_type_str, check_type_float
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _positive_float(val):
    float_val = check_type_float(val)
    if float_val < 0:
        return 0
    else:
        return float_val