from __future__ import absolute_import, division, print_function
import copy
import glob
import os
from importlib import import_module
from ansible.errors import AnsibleActionFail, AnsibleError
from ansible.module_utils._text import to_text
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.action.network import (
def _get_os_name(self):
    os_name = None
    if 'network_os' in self._task.args and self._task.args['network_os']:
        display.vvvv('Getting OS name from task argument')
        os_name = self._task.args['network_os']
    elif self._play_context.network_os:
        display.vvvv('Getting OS name from inventory')
        os_name = self._play_context.network_os
    elif 'network_os' in self._task_vars.get('ansible_facts', {}) and self._task_vars['ansible_facts']['network_os']:
        display.vvvv('Getting OS name from fact')
        os_name = self._task_vars['ansible_facts']['network_os']
    return os_name