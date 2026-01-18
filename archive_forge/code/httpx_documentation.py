from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import BAGGAGE_HEADER_NAME
from sentry_sdk.tracing_utils import should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING

        httpx has its own transport layer and can be customized when needed,
        so patch Client.send and AsyncClient.send to support both synchronous and async interfaces.
        