import sys
import contextlib
from datetime import datetime, timedelta
from functools import wraps
from sentry_sdk._types import TYPE_CHECKING
@wraps(func)
class DecoratorContextManager:

    def __init__(self, *args, **kwargs):
        self.the_contextmanager = contextmanager_func(*args, **kwargs)

    def __enter__(self):
        self.the_contextmanager.__enter__()

    def __exit__(self, *args, **kwargs):
        self.the_contextmanager.__exit__(*args, **kwargs)

    def __call__(self, decorated_func):

        @wraps(decorated_func)
        def when_called(*args, **kwargs):
            with self.the_contextmanager:
                return_val = decorated_func(*args, **kwargs)
            return return_val
        return when_called