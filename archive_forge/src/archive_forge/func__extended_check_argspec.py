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
def _extended_check_argspec(self):
    """Check additional requirements for the argspec
        that cannot be covered using stnd techniques
        """
    errors = []
    requested_parser = self._task.args.get('parser').get('name')
    if len(requested_parser.split('.')) != 3:
        msg = 'Parser name should be provided as a full name including collection'
        errors.append(msg)
    if self._task.args.get('text') and requested_parser not in ['ansible.utils.json', 'ansible.utils.xml']:
        if not (self._task.args.get('parser').get('command') or self._task.args.get('parser').get('template_path')):
            msg = 'Either parser/command or parser/template_path needs to be provided when parsing text.'
            errors.append(msg)
    if errors:
        self._result['failed'] = True
        self._result['msg'] = ' '.join(errors)