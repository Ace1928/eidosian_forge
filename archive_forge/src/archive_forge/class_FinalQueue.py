from __future__ import (absolute_import, division, print_function)
import os
import sys
import tempfile
import threading
import time
import typing as t
import multiprocessing.queues
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError
from ansible.executor.play_iterator import PlayIterator
from ansible.executor.stats import AggregateStats
from ansible.executor.task_result import TaskResult
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.playbook.play_context import PlayContext
from ansible.playbook.task import Task
from ansible.plugins.loader import callback_loader, strategy_loader, module_loader
from ansible.plugins.callback import CallbackBase
from ansible.template import Templar
from ansible.vars.hostvars import HostVars
from ansible.vars.reserved import warn_if_reserved
from ansible.utils.display import Display
from ansible.utils.lock import lock_decorator
from ansible.utils.multiprocessing import context as multiprocessing_context
from dataclasses import dataclass
class FinalQueue(multiprocessing.queues.SimpleQueue):

    def __init__(self, *args, **kwargs):
        kwargs['ctx'] = multiprocessing_context
        super().__init__(*args, **kwargs)

    def send_callback(self, method_name, *args, **kwargs):
        self.put(CallbackSend(method_name, *args, **kwargs))

    def send_task_result(self, *args, **kwargs):
        if isinstance(args[0], TaskResult):
            tr = args[0]
        else:
            tr = TaskResult(*args, **kwargs)
        self.put(tr)

    def send_display(self, method, *args, **kwargs):
        self.put(DisplaySend(method, *args, **kwargs))

    def send_prompt(self, **kwargs):
        self.put(PromptSend(**kwargs))