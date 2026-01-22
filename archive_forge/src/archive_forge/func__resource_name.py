import hashlib
from functools import cached_property
from inspect import isawaitable
from sentry_sdk import configure_scope, start_span
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
@cached_property
def _resource_name(self):
    query_hash = self.hash_query(self.execution_context.query)
    if self.execution_context.operation_name:
        return '{}:{}'.format(self.execution_context.operation_name, query_hash)
    return query_hash