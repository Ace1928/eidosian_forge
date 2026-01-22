import numbers
from billiard.exceptions import SoftTimeLimitExceeded, Terminated, TimeLimitExceeded, WorkerLostError
from click import ClickException
from kombu.exceptions import OperationalError
from celery.utils.serialization import get_pickleable_exception
class BackendStoreError(BackendError):
    """An issue writing to the backend."""

    def __init__(self, *args, **kwargs):
        self.state = kwargs.get('state', '')
        self.task_id = kwargs.get('task_id', '')

    def __repr__(self):
        return super().__repr__() + ' state:' + self.state + ' task_id:' + self.task_id