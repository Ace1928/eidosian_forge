import numbers
from billiard.exceptions import SoftTimeLimitExceeded, Terminated, TimeLimitExceeded, WorkerLostError
from click import ClickException
from kombu.exceptions import OperationalError
from celery.utils.serialization import get_pickleable_exception
class MaxRetriesExceededError(TaskError):
    """The tasks max restart limit has been exceeded."""

    def __init__(self, *args, **kwargs):
        self.task_args = kwargs.pop('task_args', [])
        self.task_kwargs = kwargs.pop('task_kwargs', dict())
        super().__init__(*args, **kwargs)