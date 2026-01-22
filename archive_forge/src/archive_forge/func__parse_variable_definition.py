from __future__ import (absolute_import, division, print_function)
import ast
import re
import warnings
from ansible.inventory.group import to_safe_group_name
from ansible.plugins.inventory import BaseFileInventoryPlugin
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.utils.shlex import shlex_split
def _parse_variable_definition(self, line):
    """
        Takes a string and tries to parse it as a variable definition. Returns
        the key and value if successful, or raises an error.
        """
    if '=' in line:
        k, v = [e.strip() for e in line.split('=', 1)]
        return (k, self._parse_value(v))
    self._raise_error('Expected key=value, got: %s' % line)