import functools
import signal
import time
from oslo_utils import importutils
from osprofiler.drivers import base
class NotifyEndpoint(object):

    def __init__(self, oslo_messaging, base_id):
        self.received_messages = []
        self.last_read_time = time.time()
        self.filter_rule = oslo_messaging.NotificationFilter(payload={'base_id': base_id})

    def info(self, ctxt, publisher_id, event_type, payload, metadata):
        self.received_messages.append(payload)
        self.last_read_time = time.time()

    def get_messages(self):
        return self.received_messages

    def get_last_read_time(self):
        return self.last_read_time