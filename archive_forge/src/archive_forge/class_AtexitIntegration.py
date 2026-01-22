from __future__ import absolute_import
import os
import sys
import atexit
from sentry_sdk.hub import Hub
from sentry_sdk.utils import logger
from sentry_sdk.integrations import Integration
from sentry_sdk._types import TYPE_CHECKING
class AtexitIntegration(Integration):
    identifier = 'atexit'

    def __init__(self, callback=None):
        if callback is None:
            callback = default_callback
        self.callback = callback

    @staticmethod
    def setup_once():

        @atexit.register
        def _shutdown():
            logger.debug('atexit: got shutdown signal')
            hub = Hub.main
            integration = hub.get_integration(AtexitIntegration)
            if integration is not None:
                logger.debug('atexit: shutting down client')
                hub.end_session()
                client = hub.client
                client.close(callback=integration.callback)