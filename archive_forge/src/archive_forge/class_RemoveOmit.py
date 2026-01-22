from __future__ import absolute_import, division, print_function
import copy
import traceback
import os
from contextlib import contextmanager
import platform
from ansible.config.manager import ensure_type
from ansible.errors import (
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types, iteritems
from ansible.module_utils._text import to_text, to_bytes, to_native
from ansible.plugins.action import ActionBase
class RemoveOmit(object):

    def __init__(self, buffer, omit_value):
        try:
            import yaml
        except ImportError:
            raise AnsibleError('Failed to import the required Python library (PyYAML).')
        self.data = yaml.safe_load_all(buffer)
        self.omit = omit_value

    def remove_omit(self, data):
        if isinstance(data, dict):
            result = dict()
            for key, value in iteritems(data):
                if value == self.omit:
                    continue
                result[key] = self.remove_omit(value)
            return result
        if isinstance(data, list):
            return [self.remove_omit(v) for v in data if v != self.omit]
        return data

    def output(self):
        return [self.remove_omit(d) for d in self.data]