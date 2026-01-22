import uuid
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
def _make_uuid(val):
    if isinstance(val, uuid.UUID):
        return val
    return uuid.UUID(val)