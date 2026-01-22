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
def _prune_result(self):
    """In the case of an error, remove stdout and stdout_lines
        this allows for easier visibility of the error message.
        In the case of an actual command error, it will be thrown
        in the module
        """
    self._result.pop('stdout', None)
    self._result.pop('stdout_lines', None)