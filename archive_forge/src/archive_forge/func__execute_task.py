import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
def _execute_task(task, arguments, progress_callback=None):
    with notifier.register_deregister(task.notifier, ta.EVENT_UPDATE_PROGRESS, callback=progress_callback):
        try:
            task.pre_execute()
            result = task.execute(**arguments)
        except Exception:
            result = failure.Failure()
        finally:
            task.post_execute()
    return (EXECUTED, result)