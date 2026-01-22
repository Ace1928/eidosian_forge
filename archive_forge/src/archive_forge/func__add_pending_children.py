from __future__ import (absolute_import, division, print_function)
import ast
import re
import warnings
from ansible.inventory.group import to_safe_group_name
from ansible.plugins.inventory import BaseFileInventoryPlugin
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.utils.shlex import shlex_split
def _add_pending_children(self, group, pending):
    for parent in pending[group]['parents']:
        self.inventory.add_child(parent, group)
        if parent in pending and pending[parent]['state'] == 'children':
            self._add_pending_children(parent, pending)
    del pending[group]