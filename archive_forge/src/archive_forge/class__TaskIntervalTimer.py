from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import time
import enum
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
class _TaskIntervalTimer(object):
    """Timer to facilitate performing multiple tasks at different intervals.

  Here's an overview of how the caller sees this class:

  >>> timer = _TaskIntervalTimer({'a': 5, 'b': 10, 'c': 3})
  >>> timer.Wait()  # sleeps 3 seconds, total time elapsed 3
  ['c']
  >>> timer.Wait()  # sleeps 2 seconds, total time elapsed 5
  ['a']
  >>> timer.Wait()  # sleeps 1 second,  total time elapsed 6
  ['c']
  >>> timer.Wait()  # sleeps 3 seconds, total time elapsed 9
  ['c']
  >>> timer.Wait()  # sleeps 1 second,  total time elapsed 10
  ['a', 'c']

  And here's how it might be used in practice:

  >>> timer = _TaskIntervalTimer({'foo': 1, 'bar': 10, 'baz': 3})
  >>> while True:
  ...   tasks = timer.Wait()
  ...   if 'foo' in tasks:
  ...     foo()
  ...   if 'bar' in tasks:
  ...     bar()
  ...   if 'baz' in tasks:
  ...     do_baz()


  Attributes:
    task_intervals: dict (hashable to int), mapping from some representation of
      a task to to the interval (in seconds) at which the task should be
      performed
  """

    def __init__(self, task_intervals):
        self.task_intervals = task_intervals
        self._time_remaining = self.task_intervals.copy()

    def Wait(self):
        """Wait for the next task(s) and return them.

    Returns:
      set, the tasks which should be performed
    """
        sleep_time = min(self._time_remaining.values())
        time.sleep(sleep_time)
        tasks = set()
        for task in self._time_remaining:
            self._time_remaining[task] -= sleep_time
            if self._time_remaining[task] == 0:
                self._time_remaining[task] = self.task_intervals[task]
                tasks.add(task)
        return tasks