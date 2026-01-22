from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.plugins.callback import CallbackBase
from ansible.utils.color import colorize, hostcolor
from ansible.playbook.task_include import TaskInclude
def _all_vars(self, host=None, task=None):
    return self._play.get_variable_manager().get_vars(play=self._play, host=host, task=task)