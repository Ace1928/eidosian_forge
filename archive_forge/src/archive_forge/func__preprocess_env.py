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
def _preprocess_env(module, values):
    if not values:
        return {}
    final_env = {}
    if 'env_file' in values:
        parsed_env_file = parse_env_file(values['env_file'])
        for name, value in parsed_env_file.items():
            final_env[name] = to_text(value, errors='surrogate_or_strict')
    if 'env' in values:
        for name, value in values['env'].items():
            if not isinstance(value, string_types):
                module.fail_json(msg='Non-string value found for env option. Ambiguous env options must be wrapped in quotes to avoid them being interpreted. Key: %s' % (name,))
            final_env[name] = to_text(value, errors='surrogate_or_strict')
    formatted_env = []
    for key, value in final_env.items():
        formatted_env.append('%s=%s' % (key, value))
    return {'env': formatted_env}