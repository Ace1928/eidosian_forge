from __future__ import (absolute_import, division, print_function)
from os.path import basename
from ansible import constants as C
from ansible import context
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.color import colorize, hostcolor
from ansible.plugins.callback.default import CallbackModule as CallbackModule_default
def _get_task_display_name(self, task):
    self.task_display_name = None
    display_name = task.get_name().strip().split(' : ')
    task_display_name = display_name[-1]
    if task_display_name.startswith('include'):
        return
    else:
        self.task_display_name = task_display_name