from vine.utils import wraps
from celery.exceptions import Ignore, Retry
from celery.utils.time import get_exponential_backoff_interval
Wrap task's `run` method with auto-retry functionality.