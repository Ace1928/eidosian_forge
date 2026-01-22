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
def _patch_schema_init():
    old_schema_init = Schema.__init__

    def _sentry_patched_schema_init(self, *args, **kwargs):
        integration = Hub.current.get_integration(StrawberryIntegration)
        if integration is None:
            return old_schema_init(self, *args, **kwargs)
        extensions = kwargs.get('extensions') or []
        if integration.async_execution is not None:
            should_use_async_extension = integration.async_execution
        else:
            should_use_async_extension = _guess_if_using_async(extensions)
            logger.info('Assuming strawberry is running %s. If not, initialize it as StrawberryIntegration(async_execution=%s).', 'async' if should_use_async_extension else 'sync', 'False' if should_use_async_extension else 'True')
        extensions = [extension for extension in extensions if extension not in (StrawberrySentryAsyncExtension, StrawberrySentrySyncExtension)]
        extensions.append(SentryAsyncExtension if should_use_async_extension else SentrySyncExtension)
        kwargs['extensions'] = extensions
        return old_schema_init(self, *args, **kwargs)
    Schema.__init__ = _sentry_patched_schema_init