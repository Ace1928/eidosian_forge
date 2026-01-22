import numbers
from billiard.exceptions import SoftTimeLimitExceeded, Terminated, TimeLimitExceeded, WorkerLostError
from click import ClickException
from kombu.exceptions import OperationalError
from celery.utils.serialization import get_pickleable_exception
class AlwaysEagerIgnored(CeleryWarning):
    """send_task ignores :setting:`task_always_eager` option."""