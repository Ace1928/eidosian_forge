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
Count the number of occurrences of a key in a given dictionary
        :arg terms: String or iterable of values to check
        :arg module_parameters: Dictionary of module parameters
        :returns: An integer that is the number of occurrences of the terms values
        in the provided dictionary.
        