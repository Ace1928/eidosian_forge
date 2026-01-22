from sentry_sdk import Hub
from sentry_sdk._types import MYPY
from sentry_sdk.consts import OP
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_CUSTOM
from sentry_sdk.utils import event_from_exception
def _find_name(self, context):
    return self._handler_call_details.method