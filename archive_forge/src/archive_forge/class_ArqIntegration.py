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
class ArqIntegration(Integration):
    identifier = 'arq'

    @staticmethod
    def setup_once():
        try:
            if isinstance(ARQ_VERSION, str):
                version = parse_version(ARQ_VERSION)
            else:
                version = ARQ_VERSION.version[:2]
        except (TypeError, ValueError):
            version = None
        if version is None:
            raise DidNotEnable('Unparsable arq version: {}'.format(ARQ_VERSION))
        if version < (0, 23):
            raise DidNotEnable('arq 0.23 or newer required.')
        patch_enqueue_job()
        patch_run_job()
        patch_create_worker()
        ignore_logger('arq.worker')