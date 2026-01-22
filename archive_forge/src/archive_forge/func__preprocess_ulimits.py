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
def _preprocess_ulimits(module, values):
    if 'ulimits' not in values:
        return values
    result = []
    for limit in values['ulimits']:
        limits = dict()
        pieces = limit.split(':')
        if len(pieces) >= 2:
            limits['Name'] = pieces[0]
            limits['Soft'] = int(pieces[1])
            limits['Hard'] = int(pieces[1])
        if len(pieces) == 3:
            limits['Hard'] = int(pieces[2])
        result.append(limits)
    return {'ulimits': result}