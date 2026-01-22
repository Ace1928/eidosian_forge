from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import binary_type, text_type
@staticmethod
def _convert_defaults_str_to_list(value):
    """ Converts array output from defaults to an list """
    value = value.splitlines()
    value.pop(0)
    value.pop(-1)
    value = [re.sub('^ *"?|"?,? *$', '', x.replace('\\"', '"')) for x in value]
    return value