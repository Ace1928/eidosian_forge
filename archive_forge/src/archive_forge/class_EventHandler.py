from __future__ import absolute_import
import logging
from fnmatch import fnmatch
from sentry_sdk.hub import Hub
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk._compat import iteritems, utc_from_timestamp
from sentry_sdk._types import TYPE_CHECKING
class EventHandler(_BaseHandler):
    """
    A logging handler that emits Sentry events for each log record

    Note that you do not have to use this class if the logging integration is enabled, which it is by default.
    """

    def emit(self, record):
        with capture_internal_exceptions():
            self.format(record)
            return self._emit(record)

    def _emit(self, record):
        if not self._can_record(record):
            return
        hub = Hub.current
        if hub.client is None:
            return
        client_options = hub.client.options
        if record.exc_info and record.exc_info[0] is not None:
            event, hint = event_from_exception(record.exc_info, client_options=client_options, mechanism={'type': 'logging', 'handled': True})
        elif record.exc_info and record.exc_info[0] is None:
            event = {}
            hint = {}
            with capture_internal_exceptions():
                event['threads'] = {'values': [{'stacktrace': current_stacktrace(include_local_variables=client_options['include_local_variables'], max_value_length=client_options['max_value_length']), 'crashed': False, 'current': True}]}
        else:
            event = {}
            hint = {}
        hint['log_record'] = record
        level = self._logging_to_event_level(record)
        if level in {'debug', 'info', 'warning', 'error', 'critical', 'fatal'}:
            event['level'] = level
        event['logger'] = record.name
        record_caputured_from_warnings_module = record.name == 'py.warnings' and record.msg == '%s'
        if record_caputured_from_warnings_module:
            msg = record.args[0]
            event['logentry'] = {'message': msg, 'params': ()}
        else:
            event['logentry'] = {'message': to_string(record.msg), 'params': record.args}
        event['extra'] = self._extra_from_record(record)
        hub.capture_event(event, hint=hint)