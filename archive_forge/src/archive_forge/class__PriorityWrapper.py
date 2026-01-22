from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from six.moves import queue
class _PriorityWrapper:
    """Wraps a buffered task and tracks priority information.

  Attributes:
    task (Union[task.Task, str]): A buffered item. Expected to be a task or a
      string (to handle shutdowns) when used by task_graph_executor.
    priority (int): The priority of this task. A task with a lower value will be
      executed before a task with a higher value, since queue.PriorityQueue uses
      a min-heap.
  """

    def __init__(self, task, priority):
        self.task = task
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority