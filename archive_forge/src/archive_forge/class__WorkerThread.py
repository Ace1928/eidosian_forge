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
class _WorkerThread(threading.Thread):

    def __init__(self, work_queue, results_map):
        super(_WorkerThread, self).__init__()
        self.work_queue = work_queue
        self.results_map = results_map

    def run(self):
        while True:
            thread_task = self.work_queue.get()
            if thread_task is _STOP_WORKING:
                return
            task = thread_task.task
            try:
                result = _Result((task.func(*task.args),))
            except:
                result = _Result(exc_info=sys.exc_info())
            self.results_map[thread_task.task] = result