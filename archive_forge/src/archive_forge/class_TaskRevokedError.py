import numbers
from billiard.exceptions import SoftTimeLimitExceeded, Terminated, TimeLimitExceeded, WorkerLostError
from click import ClickException
from kombu.exceptions import OperationalError
from celery.utils.serialization import get_pickleable_exception
class TaskRevokedError(TaskError):
    """The task has been revoked, so no result available."""