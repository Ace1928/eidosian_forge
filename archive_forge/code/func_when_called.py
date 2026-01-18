import sys
import contextlib
from datetime import datetime, timedelta
from functools import wraps
from sentry_sdk._types import TYPE_CHECKING
@wraps(decorated_func)
def when_called(*args, **kwargs):
    with self.the_contextmanager:
        return_val = decorated_func(*args, **kwargs)
    return return_val