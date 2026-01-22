from copy import copy
from collections import deque
from itertools import chain
import os
import sys
import uuid
from sentry_sdk.attachments import Attachment
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk.consts import FALSE_VALUES, INSTRUMENTER
from sentry_sdk._functools import wraps
from sentry_sdk.profiler import Profile
from sentry_sdk.session import Session
from sentry_sdk.tracing_utils import (
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def add_error_processor(self, func, cls=None):
    """Register a scope local error processor on the scope.

        :param func: A callback that works similar to an event processor but is invoked with the original exception info triple as second argument.

        :param cls: Optionally, only process exceptions of this type.
        """
    if cls is not None:
        cls_ = cls
        real_func = func

        def func(event, exc_info):
            try:
                is_inst = isinstance(exc_info[1], cls_)
            except Exception:
                is_inst = False
            if is_inst:
                return real_func(event, exc_info)
            return event
    self._error_processors.append(func)