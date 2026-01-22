from __future__ import absolute_import, division, print_function
import json
from importlib import import_module
from ansible.errors import AnsibleActionFail
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible.module_utils.connection import ConnectionError as AnsibleConnectionError
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.modules.cli_parse import DOCUMENTATION
def _set_parser_command(self):
    """Set the /parser/command in the task args based on /command if needed"""
    if self._task.args.get('command'):
        if not self._task.args.get('parser').get('command'):
            self._task.args.get('parser')['command'] = self._task.args.get('command')