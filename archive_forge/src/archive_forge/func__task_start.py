from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.playbook.task_include import TaskInclude
from ansible.plugins.callback import CallbackBase
from ansible.utils.color import colorize, hostcolor
from ansible.utils.fqcn import add_internal_fqcns
def _task_start(self, task, prefix=None):
    if prefix is not None:
        self._task_type_cache[task._uuid] = prefix
    if self._play.strategy in add_internal_fqcns(('free', 'host_pinned')):
        self._last_task_name = None
    else:
        self._last_task_name = task.get_name().strip()
        if self.get_option('display_skipped_hosts') and self.get_option('display_ok_hosts'):
            self._print_task_banner(task)