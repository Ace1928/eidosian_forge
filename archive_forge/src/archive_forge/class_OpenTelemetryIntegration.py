import sys
from importlib import import_module
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.opentelemetry.span_processor import SentrySpanProcessor
from sentry_sdk.integrations.opentelemetry.propagator import SentryPropagator
from sentry_sdk.utils import logger, _get_installed_modules
from sentry_sdk._types import TYPE_CHECKING
class OpenTelemetryIntegration(Integration):
    identifier = 'opentelemetry'

    @staticmethod
    def setup_once():
        logger.warning('[OTel] Initializing highly experimental OpenTelemetry support. Use at your own risk.')
        original_classes = _record_unpatched_classes()
        try:
            distro = _load_distro()
            distro.configure()
            _load_instrumentors(distro)
        except Exception:
            logger.exception('[OTel] Failed to auto-initialize OpenTelemetry')
        try:
            _patch_remaining_classes(original_classes)
        except Exception:
            logger.exception('[OTel] Failed to post-patch instrumented classes. You might have to make sure sentry_sdk.init() is called before importing anything else.')
        _setup_sentry_tracing()
        logger.debug('[OTel] Finished setting up OpenTelemetry integration')