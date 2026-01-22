from __future__ import absolute_import
import sys
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import Hub
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
def _get_arq_function(func):
    arq_func = arq.worker.func(func)
    arq_func.coroutine = _wrap_coroutine(arq_func.name, arq_func.coroutine)
    return arq_func