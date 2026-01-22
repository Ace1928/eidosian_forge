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
def _load_trace_data_from_env(self):
    """
        Load Sentry trace id and baggage from environment variables.
        Can be disabled by setting SENTRY_USE_ENVIRONMENT to "false".
        """
    incoming_trace_information = None
    sentry_use_environment = (os.environ.get('SENTRY_USE_ENVIRONMENT') or '').lower()
    use_environment = sentry_use_environment not in FALSE_VALUES
    if use_environment:
        incoming_trace_information = {}
        if os.environ.get('SENTRY_TRACE'):
            incoming_trace_information[SENTRY_TRACE_HEADER_NAME] = os.environ.get('SENTRY_TRACE') or ''
        if os.environ.get('SENTRY_BAGGAGE'):
            incoming_trace_information[BAGGAGE_HEADER_NAME] = os.environ.get('SENTRY_BAGGAGE') or ''
    return incoming_trace_information or None