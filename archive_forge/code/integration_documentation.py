import sys
from importlib import import_module
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.opentelemetry.span_processor import SentrySpanProcessor
from sentry_sdk.integrations.opentelemetry.propagator import SentryPropagator
from sentry_sdk.utils import logger, _get_installed_modules
from sentry_sdk._types import TYPE_CHECKING

    Best-effort attempt to patch any uninstrumented classes in sys.modules.

    This enables us to not care about the order of imports and sentry_sdk.init()
    in user code. If e.g. the Flask class had been imported before sentry_sdk
    was init()ed (and therefore before the OTel instrumentation ran), it would
    not be instrumented. This function goes over remaining uninstrumented
    occurrences of the class in sys.modules and replaces them with the
    instrumented class.

    Since this is looking for exact matches, it will not work in some scenarios
    (e.g. if someone is not using the specific class explicitly, but rather
    inheriting from it). In those cases it's still necessary to sentry_sdk.init()
    before importing anything that's supposed to be instrumented.
    