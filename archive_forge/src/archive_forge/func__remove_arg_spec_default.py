from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
def _remove_arg_spec_default(self, data):
    """Used to remove any data keys that were not provided by user, but has the arg spec
        default values
        """
    new_dict = dict()
    for k, v in data.items():
        if isinstance(v, dict):
            v = self._remove_arg_spec_default(v)
            new_dict[k] = v
        elif v is not None:
            new_dict[k] = v
    return new_dict