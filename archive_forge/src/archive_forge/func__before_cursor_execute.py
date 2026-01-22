from __future__ import absolute_import
from sentry_sdk._compat import text_type
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import SPANDATA
from sentry_sdk.db.explain_plan.sqlalchemy import attach_explain_plan_to_span
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import capture_internal_exceptions, parse_version
def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany, *args):
    hub = Hub.current
    if hub.get_integration(SqlalchemyIntegration) is None:
        return
    ctx_mgr = record_sql_queries(hub, cursor, statement, parameters, paramstyle=context and context.dialect and context.dialect.paramstyle or None, executemany=executemany)
    context._sentry_sql_span_manager = ctx_mgr
    span = ctx_mgr.__enter__()
    if span is not None:
        _set_db_data(span, conn)
        if hub.client:
            options = hub.client.options['_experiments'].get('attach_explain_plans')
            if options is not None:
                attach_explain_plan_to_span(span, conn, statement, parameters, options)
        context._sentry_sql_span = span