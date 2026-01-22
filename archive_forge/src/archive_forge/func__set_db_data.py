from __future__ import absolute_import
import inspect
import sys
import threading
import weakref
from importlib import import_module
from sentry_sdk._compat import string_types, text_type
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.db.explain_plan.django import attach_explain_plan_to_span
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.serializer import add_global_repr_processor
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_URL
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.django.transactions import LEGACY_RESOLVER
from sentry_sdk.integrations.django.templates import (
from sentry_sdk.integrations.django.middleware import patch_django_middlewares
from sentry_sdk.integrations.django.signals_handlers import patch_signals
from sentry_sdk.integrations.django.views import patch_views
def _set_db_data(span, cursor_or_db):
    db = cursor_or_db.db if hasattr(cursor_or_db, 'db') else cursor_or_db
    vendor = db.vendor
    span.set_data(SPANDATA.DB_SYSTEM, vendor)
    is_psycopg2 = hasattr(cursor_or_db, 'connection') and hasattr(cursor_or_db.connection, 'get_dsn_parameters') and inspect.isroutine(cursor_or_db.connection.get_dsn_parameters)
    if is_psycopg2:
        connection_params = cursor_or_db.connection.get_dsn_parameters()
    else:
        is_psycopg3 = hasattr(cursor_or_db, 'connection') and hasattr(cursor_or_db.connection, 'info') and hasattr(cursor_or_db.connection.info, 'get_parameters') and inspect.isroutine(cursor_or_db.connection.info.get_parameters)
        if is_psycopg3:
            connection_params = cursor_or_db.connection.info.get_parameters()
        else:
            connection_params = db.get_connection_params()
    db_name = connection_params.get('dbname') or connection_params.get('database')
    if db_name is not None:
        span.set_data(SPANDATA.DB_NAME, db_name)
    server_address = connection_params.get('host')
    if server_address is not None:
        span.set_data(SPANDATA.SERVER_ADDRESS, server_address)
    server_port = connection_params.get('port')
    if server_port is not None:
        span.set_data(SPANDATA.SERVER_PORT, text_type(server_port))
    server_socket_address = connection_params.get('unix_socket')
    if server_socket_address is not None:
        span.set_data(SPANDATA.SERVER_SOCKET_ADDRESS, server_socket_address)