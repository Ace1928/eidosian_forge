from __future__ import absolute_import, division, print_function
import abc
import os
import re
import shlex
from functools import partial
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _preprocess_log(module, values):
    result = {}
    if 'log_driver' not in values:
        return result
    result['log_driver'] = values['log_driver']
    if 'log_options' in values:
        options = {}
        for k, v in values['log_options'].items():
            if not isinstance(v, string_types):
                module.warn("Non-string value found for log_options option '%s'. The value is automatically converted to '%s'. If this is not correct, or you want to avoid such warnings, please quote the value." % (k, to_text(v, errors='surrogate_or_strict')))
            v = to_text(v, errors='surrogate_or_strict')
            options[k] = v
        result['log_options'] = options
    return result