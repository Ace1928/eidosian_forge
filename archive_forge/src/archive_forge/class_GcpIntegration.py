import sys
from copy import deepcopy
from datetime import timedelta
from os import environ
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk._compat import datetime_utcnow, duration_in_milliseconds, reraise
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._types import TYPE_CHECKING
class GcpIntegration(Integration):
    identifier = 'gcp'

    def __init__(self, timeout_warning=False):
        self.timeout_warning = timeout_warning

    @staticmethod
    def setup_once():
        import __main__ as gcp_functions
        if not hasattr(gcp_functions, 'worker_v1'):
            logger.warning('GcpIntegration currently supports only Python 3.7 runtime environment.')
            return
        worker1 = gcp_functions.worker_v1
        worker1.FunctionHandler.invoke_user_function = _wrap_func(worker1.FunctionHandler.invoke_user_function)