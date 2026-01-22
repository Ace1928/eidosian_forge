from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import pickle
import sys
import threading
import time
from googlecloudsdk.core import exceptions
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import queue   # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def ApplyAsync(self, func, args):
    if not self.worker_threads:
        raise InvalidStateException('ThreadPool must be Start()ed before use.')
    task = _Task(func, args)
    result = _ThreadFuture(task, self._results_map)
    self._task_queue.put(_ThreadTask(task))
    return result