from __future__ import (absolute_import, division, print_function)
import csv
import datetime
import os
import time
import threading
from abc import ABCMeta, abstractmethod
from functools import partial
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible.parsing.ajson import AnsibleJSONEncoder, json
from ansible.plugins.callback import CallbackBase
def _profile(self, obj=None):
    prev_task = None
    results = dict.fromkeys(self._features)
    if not obj or self._file_per_task:
        for dummy, f in self._files.items():
            if f is None:
                continue
            try:
                f.close()
            except Exception:
                pass
    try:
        for name, prof in self._profilers.items():
            prof.running = False
        for name, prof in self._profilers.items():
            results[name] = prof.max
        prev_task = prof.obj
    except AttributeError:
        pass
    for name, result in results.items():
        if result is not None:
            try:
                self.task_results[name].append((prev_task, result))
            except ValueError:
                pass
    if obj is not None:
        if self._file_per_task or self._counter == 0:
            self._open_files(task_uuid=obj._uuid)
        for feature in self._features:
            self._profilers[feature] = self._profiler_map[feature](obj=obj, writer=self._writers[feature])
            self._profilers[feature].start()
        self._counter += 1