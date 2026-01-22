from __future__ import absolute_import
from sentry_sdk._compat import text_type
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import SPANDATA
from sentry_sdk.db.explain_plan.sqlalchemy import attach_explain_plan_to_span
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import capture_internal_exceptions, parse_version
def _after_cursor_execute(conn, cursor, statement, parameters, context, *args):
    hub = Hub.current
    if hub.get_integration(SqlalchemyIntegration) is None:
        return
    ctx_mgr = getattr(context, '_sentry_sql_span_manager', None)
    if ctx_mgr is not None:
        context._sentry_sql_span_manager = None
        ctx_mgr.__exit__(None, None, None)
    span = getattr(context, '_sentry_sql_span', None)
    if span is not None:
        with capture_internal_exceptions():
            add_query_source(hub, span)