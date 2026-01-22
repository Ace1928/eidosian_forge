from contextlib import ContextDecorator, contextmanager
from django.db import (
def _non_atomic_requests(view, using):
    try:
        view._non_atomic_requests.add(using)
    except AttributeError:
        view._non_atomic_requests = {using}
    return view