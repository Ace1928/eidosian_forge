import weakref
from taskflow import exceptions as excp
from taskflow import states as st
from taskflow.types import failure
class TaskScheduler(object):
    """Schedules task atoms."""

    def __init__(self, runtime):
        self._storage = runtime.storage
        self._task_action = runtime.task_action

    def schedule(self, task):
        """Schedules the given task atom for *future* completion.

        Depending on the atoms stored intention this may schedule the task
        atom for reversion or execution.
        """
        intention = self._storage.get_atom_intention(task.name)
        if intention == st.EXECUTE:
            return self._task_action.schedule_execution(task)
        elif intention == st.REVERT:
            return self._task_action.schedule_reversion(task)
        else:
            raise excp.ExecutionFailure('Unknown how to schedule task with intention: %s' % intention)