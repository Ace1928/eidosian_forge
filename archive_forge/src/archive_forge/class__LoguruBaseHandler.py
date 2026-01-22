from __future__ import absolute_import
import enum
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import (
class _LoguruBaseHandler(_BaseHandler):

    def _logging_to_event_level(self, record):
        try:
            return LoggingLevels(record.levelno).name.lower()
        except ValueError:
            return record.levelname.lower() if record.levelname else ''