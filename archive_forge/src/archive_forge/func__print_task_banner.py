from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.playbook.task_include import TaskInclude
from ansible.plugins.callback import CallbackBase
from ansible.utils.color import colorize, hostcolor
from ansible.utils.fqcn import add_internal_fqcns
def _print_task_banner(self, task):
    args = ''
    if not task.no_log and C.DISPLAY_ARGS_TO_STDOUT:
        args = u', '.join((u'%s=%s' % a for a in task.args.items()))
        args = u' %s' % args
    prefix = self._task_type_cache.get(task._uuid, 'TASK')
    task_name = self._last_task_name
    if task_name is None:
        task_name = task.get_name().strip()
    if task.check_mode and self.get_option('check_mode_markers'):
        checkmsg = ' [CHECK MODE]'
    else:
        checkmsg = ''
    self._display.banner(u'%s [%s%s]%s' % (prefix, task_name, args, checkmsg))
    if self._display.verbosity >= 2:
        self._print_task_path(task)
    self._last_task_banner = task._uuid