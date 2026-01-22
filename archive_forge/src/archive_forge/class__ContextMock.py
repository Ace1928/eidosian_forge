import numbers
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence  # noqa
from unittest.mock import Mock
from celery import Celery  # noqa
from celery.canvas import Signature  # noqa
class _ContextMock(Mock):
    """Dummy class implementing __enter__ and __exit__.

    The :keyword:`with` statement requires these to be implemented
    in the class, not just the instance.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass