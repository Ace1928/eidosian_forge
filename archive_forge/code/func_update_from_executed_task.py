from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
def update_from_executed_task(self, executed_task_wrapper, task_output):
    """Updates the graph based on the output of an executed task.

    If some googlecloudsdk.command_lib.storage.task.Task instance `a` returns
    the following iterables of tasks: [[b, c], [d, e]], we need to update the
    graph as follows to ensure they are executed appropriately.

           /-- d <-\\--/- b
      a <-/         \\/
          \\         /\\
           \\-- e <-/--\\- c

    After making these updates, `b` and `c` are ready for submission. If a task
    does not return any new tasks, then it will be removed from the graph,
    potentially freeing up tasks that depend on it for execution.

    See go/parallel-processing-in-gcloud-storage#heading=h.y4o7a9hcs89r for a
    more thorough description of the updates this method performs.

    Args:
      executed_task_wrapper (task_graph.TaskWrapper): Contains information about
        how a completed task fits into a dependency graph.
      task_output (Optional[task.Output]): Additional tasks and
        messages returned by the task in executed_task_wrapper.

    Returns:
      An Iterable[task_graph.TaskWrapper] containing tasks that are ready to be
      executed after performing graph updates.
    """
    with self._lock:
        if task_output is not None and task_output.messages is not None and (executed_task_wrapper.dependent_task_ids is not None):
            for task_id in executed_task_wrapper.dependent_task_ids:
                dependent_task_wrapper = self._task_wrappers_in_graph[task_id]
                dependent_task_wrapper.task.received_messages.extend(task_output.messages)
        if task_output is None or not task_output.additional_task_iterators:
            return self.complete(executed_task_wrapper)
        parent_tasks_for_next_layer = [executed_task_wrapper]
        for task_iterator in reversed(task_output.additional_task_iterators):
            dependent_task_ids = [task_wrapper.id for task_wrapper in parent_tasks_for_next_layer]
            parent_tasks_for_next_layer = [self.add(task, dependent_task_ids=dependent_task_ids) for task in task_iterator]
        return parent_tasks_for_next_layer