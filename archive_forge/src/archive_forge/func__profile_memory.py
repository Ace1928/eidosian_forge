from __future__ import (absolute_import, division, print_function)
import time
import threading
from ansible.plugins.callback import CallbackBase
def _profile_memory(self, obj=None):
    prev_task = None
    results = None
    try:
        self._task_memprof.running = False
        results = self._task_memprof.results
        prev_task = self._task_memprof.obj
    except AttributeError:
        pass
    if obj is not None:
        self._task_memprof = MemProf(self.cgroup_current_file, obj=obj)
        self._task_memprof.start()
    if results is not None:
        self.task_results.append((prev_task, max(results)))