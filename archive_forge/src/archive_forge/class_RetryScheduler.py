import weakref
from taskflow import exceptions as excp
from taskflow import states as st
from taskflow.types import failure
class RetryScheduler(object):
    """Schedules retry atoms."""

    def __init__(self, runtime):
        self._runtime = weakref.proxy(runtime)
        self._retry_action = runtime.retry_action
        self._storage = runtime.storage

    def schedule(self, retry):
        """Schedules the given retry atom for *future* completion.

        Depending on the atoms stored intention this may schedule the retry
        atom for reversion or execution.
        """
        intention = self._storage.get_atom_intention(retry.name)
        if intention == st.EXECUTE:
            return self._retry_action.schedule_execution(retry)
        elif intention == st.REVERT:
            return self._retry_action.schedule_reversion(retry)
        elif intention == st.RETRY:
            self._retry_action.change_state(retry, st.RETRYING)
            self._runtime.retry_subflow(retry)
            return self._retry_action.schedule_execution(retry)
        else:
            raise excp.ExecutionFailure('Unknown how to schedule retry with intention: %s' % intention)