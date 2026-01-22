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
class SentrySyncExtension(SentryAsyncExtension):

    def resolve(self, _next, root, info, *args, **kwargs):
        if self.should_skip_tracing(_next, info):
            return _next(root, info, *args, **kwargs)
        field_path = '{}.{}'.format(info.parent_type, info.field_name)
        with self.graphql_span.start_child(op=OP.GRAPHQL_RESOLVE, description='resolving {}'.format(field_path)) as span:
            span.set_data('graphql.field_name', info.field_name)
            span.set_data('graphql.parent_type', info.parent_type.name)
            span.set_data('graphql.field_path', field_path)
            span.set_data('graphql.path', '.'.join(map(str, info.path.as_list())))
            return _next(root, info, *args, **kwargs)