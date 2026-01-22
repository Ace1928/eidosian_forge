import numbers
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence  # noqa
from unittest.mock import Mock
from celery import Celery  # noqa
from celery.canvas import Signature  # noqa
def ContextMock(*args, **kwargs):
    """Mock that mocks :keyword:`with` statement contexts."""
    obj = _ContextMock(*args, **kwargs)
    obj.attach_mock(_ContextMock(), '__enter__')
    obj.attach_mock(_ContextMock(), '__exit__')
    obj.__enter__.return_value = obj
    obj.__exit__.return_value = None
    return obj