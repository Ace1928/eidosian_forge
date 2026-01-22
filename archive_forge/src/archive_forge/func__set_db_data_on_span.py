from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _set_db_data_on_span(span, connection_params):
    span.set_data(SPANDATA.DB_SYSTEM, 'redis')
    db = connection_params.get('db')
    if db is not None:
        span.set_data(SPANDATA.DB_NAME, text_type(db))
    host = connection_params.get('host')
    if host is not None:
        span.set_data(SPANDATA.SERVER_ADDRESS, host)
    port = connection_params.get('port')
    if port is not None:
        span.set_data(SPANDATA.SERVER_PORT, port)