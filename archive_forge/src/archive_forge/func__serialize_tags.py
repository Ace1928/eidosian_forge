import io
import os
import random
import re
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from datetime import datetime
from functools import wraps, partial
import sentry_sdk
from sentry_sdk._compat import text_type, utc_from_timestamp, iteritems
from sentry_sdk.utils import (
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
def _serialize_tags(tags):
    if not tags:
        return ()
    rv = []
    for key, value in iteritems(tags):
        if isinstance(value, (list, tuple)):
            for inner_value in value:
                if inner_value is not None:
                    rv.append((key, text_type(inner_value)))
        elif value is not None:
            rv.append((key, text_type(value)))
    return tuple(sorted(rv))