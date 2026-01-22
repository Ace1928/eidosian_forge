from __future__ import absolute_import
import sys
from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _sentry_worker_main(*args, **kwargs):
    import pyspark.worker as original_worker
    try:
        original_worker.main(*args, **kwargs)
    except SystemExit:
        if Hub.current.get_integration(SparkWorkerIntegration) is not None:
            hub = Hub.current
            exc_info = sys.exc_info()
            with capture_internal_exceptions():
                _capture_exception(exc_info, hub)