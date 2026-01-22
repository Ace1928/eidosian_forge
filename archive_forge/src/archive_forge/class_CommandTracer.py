from __future__ import absolute_import
import copy
from sentry_sdk import Hub
from sentry_sdk.consts import SPANDATA
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import Span
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
class CommandTracer(monitoring.CommandListener):

    def __init__(self):
        self._ongoing_operations = {}

    def _operation_key(self, event):
        return event.request_id

    def started(self, event):
        hub = Hub.current
        if hub.get_integration(PyMongoIntegration) is None:
            return
        with capture_internal_exceptions():
            command = dict(copy.deepcopy(event.command))
            command.pop('$db', None)
            command.pop('$clusterTime', None)
            command.pop('$signature', None)
            op = 'db.query'
            tags = {'db.name': event.database_name, SPANDATA.DB_SYSTEM: 'mongodb', SPANDATA.DB_OPERATION: event.command_name}
            try:
                tags['net.peer.name'] = event.connection_id[0]
                tags['net.peer.port'] = str(event.connection_id[1])
            except TypeError:
                pass
            data = {'operation_ids': {}}
            data['operation_ids']['operation'] = event.operation_id
            data['operation_ids']['request'] = event.request_id
            data.update(_get_db_data(event))
            try:
                lsid = command.pop('lsid')['id']
                data['operation_ids']['session'] = str(lsid)
            except KeyError:
                pass
            if not _should_send_default_pii():
                command = _strip_pii(command)
            query = '{} {}'.format(event.command_name, command)
            span = hub.start_span(op=op, description=query)
            for tag, value in tags.items():
                span.set_tag(tag, value)
            for key, value in data.items():
                span.set_data(key, value)
            with capture_internal_exceptions():
                hub.add_breadcrumb(message=query, category='query', type=op, data=tags)
            self._ongoing_operations[self._operation_key(event)] = span.__enter__()

    def failed(self, event):
        hub = Hub.current
        if hub.get_integration(PyMongoIntegration) is None:
            return
        try:
            span = self._ongoing_operations.pop(self._operation_key(event))
            span.set_status('internal_error')
            span.__exit__(None, None, None)
        except KeyError:
            return

    def succeeded(self, event):
        hub = Hub.current
        if hub.get_integration(PyMongoIntegration) is None:
            return
        try:
            span = self._ongoing_operations.pop(self._operation_key(event))
            span.set_status('ok')
            span.__exit__(None, None, None)
        except KeyError:
            pass