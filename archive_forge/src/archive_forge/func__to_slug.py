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
def _to_slug(self, value):
    """
        :returns slug (str): Slugified value
        :params value (str): Value that needs to be changed to slug format
        """
    if value is None:
        return value
    elif isinstance(value, int):
        return value
    else:
        removed_chars = re.sub('[^\\-\\.\\w\\s]', '', value)
        convert_chars = re.sub('[\\-\\.\\s]+', '-', removed_chars)
        return convert_chars.strip().lower()