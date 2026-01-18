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
@transaction.setter
def transaction(self, value):
    """When set this forces a specific transaction name to be set.

        Deprecated: use set_transaction_name instead."""
    logger.warning('Assigning to scope.transaction directly is deprecated: use scope.set_transaction_name() instead.')
    self._transaction = value
    if self._span and self._span.containing_transaction:
        self._span.containing_transaction.name = value