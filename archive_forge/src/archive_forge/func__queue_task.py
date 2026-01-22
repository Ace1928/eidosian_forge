from __future__ import (absolute_import, division, print_function)
import cmd
import functools
import os
import pprint
import queue
import sys
import threading
import time
import typing as t
from collections import deque
from multiprocessing import Lock
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleUndefinedVariable, AnsibleParserError
from ansible.executor import action_write_locks
from ansible.executor.play_iterator import IteratingStates, PlayIterator
from ansible.executor.process.worker import WorkerProcess
from ansible.executor.task_result import TaskResult
from ansible.executor.task_queue_manager import CallbackSend, DisplaySend, PromptSend
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.playbook.conditional import Conditional
from ansible.playbook.handler import Handler
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.task import Task
from ansible.playbook.task_include import TaskInclude
from ansible.plugins import loader as plugin_loader
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.fqcn import add_internal_fqcns
from ansible.utils.unsafe_proxy import wrap_var
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier
from ansible.vars.clean import strip_internal_keys, module_response_deepcopy
def _queue_task(self, host, task, task_vars, play_context):
    """ handles queueing the task up to be sent to a worker """
    display.debug('entering _queue_task() for %s/%s' % (host.name, task.action))
    if task.action not in action_write_locks.action_write_locks:
        display.debug('Creating lock for %s' % task.action)
        action_write_locks.action_write_locks[task.action] = Lock()
    templar = Templar(loader=self._loader, variables=task_vars)
    try:
        throttle = int(templar.template(task.throttle))
    except Exception as e:
        raise AnsibleError('Failed to convert the throttle value to an integer.', obj=task._ds, orig_exc=e)
    try:
        rewind_point = len(self._workers)
        if throttle > 0 and self.ALLOW_BASE_THROTTLING:
            if task.run_once:
                display.debug("Ignoring 'throttle' as 'run_once' is also set for '%s'" % task.get_name())
            elif throttle <= rewind_point:
                display.debug('task: %s, throttle: %d' % (task.get_name(), throttle))
                rewind_point = throttle
        queued = False
        starting_worker = self._cur_worker
        while True:
            if self._cur_worker >= rewind_point:
                self._cur_worker = 0
            worker_prc = self._workers[self._cur_worker]
            if worker_prc is None or not worker_prc.is_alive():
                self._queued_task_cache[host.name, task._uuid] = {'host': host, 'task': task, 'task_vars': task_vars, 'play_context': play_context}
                worker_prc = WorkerProcess(self._final_q, task_vars, host, task, play_context, self._loader, self._variable_manager, plugin_loader, self._cur_worker)
                self._workers[self._cur_worker] = worker_prc
                self._tqm.send_callback('v2_runner_on_start', host, task)
                worker_prc.start()
                display.debug('worker is %d (out of %d available)' % (self._cur_worker + 1, len(self._workers)))
                queued = True
            self._cur_worker += 1
            if self._cur_worker >= rewind_point:
                self._cur_worker = 0
            if queued:
                break
            elif self._cur_worker == starting_worker:
                time.sleep(0.0001)
        self._pending_results += 1
    except (EOFError, IOError, AssertionError) as e:
        display.debug('got an error while queuing: %s' % e)
        return
    display.debug('exiting _queue_task() for %s/%s' % (host.name, task.action))