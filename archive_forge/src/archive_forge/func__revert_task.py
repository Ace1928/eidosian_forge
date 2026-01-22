import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
def _revert_task(task, arguments, result, failures, progress_callback=None):
    arguments = arguments.copy()
    arguments[ta.REVERT_RESULT] = result
    arguments[ta.REVERT_FLOW_FAILURES] = failures
    with notifier.register_deregister(task.notifier, ta.EVENT_UPDATE_PROGRESS, callback=progress_callback):
        try:
            task.pre_revert()
            result = task.revert(**arguments)
        except Exception:
            result = failure.Failure()
        finally:
            task.post_revert()
    return (REVERTED, result)