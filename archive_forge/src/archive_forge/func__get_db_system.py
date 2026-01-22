from __future__ import absolute_import
from sentry_sdk._compat import text_type
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import SPANDATA
from sentry_sdk.db.explain_plan.sqlalchemy import attach_explain_plan_to_span
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import capture_internal_exceptions, parse_version
def _get_db_system(name):
    name = text_type(name)
    if 'sqlite' in name:
        return 'sqlite'
    if 'postgres' in name:
        return 'postgresql'
    if 'mariadb' in name:
        return 'mariadb'
    if 'mysql' in name:
        return 'mysql'
    if 'oracle' in name:
        return 'oracle'
    return None