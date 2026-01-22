from __future__ import print_function
import io
import gzip
import time
from datetime import timedelta
from collections import defaultdict
import urllib3
import certifi
from sentry_sdk.utils import Dsn, logger, capture_internal_exceptions, json_dumps
from sentry_sdk.worker import BackgroundWorker
from sentry_sdk.envelope import Envelope, Item, PayloadRef
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
def _parse_rate_limits(header, now=None):
    if now is None:
        now = datetime_utcnow()
    for limit in header.split(','):
        try:
            retry_after, categories, _ = limit.strip().split(':', 2)
            retry_after = now + timedelta(seconds=int(retry_after))
            for category in categories and categories.split(';') or (None,):
                yield (category, retry_after)
        except (LookupError, ValueError):
            continue