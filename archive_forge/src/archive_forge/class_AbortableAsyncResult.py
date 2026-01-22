from celery import Task
from celery.result import AsyncResult
class AbortableAsyncResult(AsyncResult):
    """Represents an abortable result.

    Specifically, this gives the `AsyncResult` a :meth:`abort()` method,
    that sets the state of the underlying Task to `'ABORTED'`.
    """

    def is_aborted(self):
        """Return :const:`True` if the task is (being) aborted."""
        return self.state == ABORTED

    def abort(self):
        """Set the state of the task to :const:`ABORTED`.

        Abortable tasks monitor their state at regular intervals and
        terminate execution if so.

        Warning:
            Be aware that invoking this method does not guarantee when the
            task will be aborted (or even if the task will be aborted at all).
        """
        return self.backend.store_result(self.id, result=None, state=ABORTED, traceback=None)