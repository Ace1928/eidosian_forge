from cryptography import exceptions as crypto_exception
import glance_store as store
from unittest import mock
import urllib
from oslo_config import cfg
from oslo_policy import policy
from glance.async_.flows._internal_plugins import base_download
from glance.common import exception
from glance.common import store_utils
from glance.common import wsgi
import glance.context
import glance.db.simple.api as simple_db
class FakeNotifier(object):

    def __init__(self, *_args, **kwargs):
        self.log = []

    def _notify(self, event_type, payload, level):
        log = {'notification_type': level, 'event_type': event_type, 'payload': payload}
        self.log.append(log)

    def warn(self, event_type, payload):
        self._notify(event_type, payload, 'WARN')

    def info(self, event_type, payload):
        self._notify(event_type, payload, 'INFO')

    def error(self, event_type, payload):
        self._notify(event_type, payload, 'ERROR')

    def debug(self, event_type, payload):
        self._notify(event_type, payload, 'DEBUG')

    def critical(self, event_type, payload):
        self._notify(event_type, payload, 'CRITICAL')

    def get_logs(self):
        return self.log