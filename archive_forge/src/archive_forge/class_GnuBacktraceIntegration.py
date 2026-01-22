import re
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
class GnuBacktraceIntegration(Integration):
    identifier = 'gnu_backtrace'

    @staticmethod
    def setup_once():

        @add_global_event_processor
        def process_gnu_backtrace(event, hint):
            with capture_internal_exceptions():
                return _process_gnu_backtrace(event, hint)