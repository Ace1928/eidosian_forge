from __future__ import absolute_import
import sys
from sentry_sdk._compat import reraise
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import event_from_exception
class AsyncioIntegration(Integration):
    identifier = 'asyncio'

    @staticmethod
    def setup_once():
        patch_asyncio()