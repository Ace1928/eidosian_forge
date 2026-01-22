import os
import queue
import time
import uuid
import fixtures
from oslo_config import cfg
import oslo_messaging
from oslo_messaging._drivers.kafka_driver import kafka_options
from oslo_messaging.notify import notifier
from oslo_messaging.tests import utils as test_utils
class InvalidDistribution(object):

    def __init__(self, original, received):
        self.original = original
        self.received = received
        self.missing = []
        self.extra = []
        self.wrong_order = []

    def describe(self):
        text = 'Sent %s, got %s; ' % (self.original, self.received)
        e1 = ['%r was missing' % m for m in self.missing]
        e2 = ['%r was not expected' % m for m in self.extra]
        e3 = ['%r expected before %r' % (m[0], m[1]) for m in self.wrong_order]
        return text + ', '.join(e1 + e2 + e3)

    def __len__(self):
        return len(self.extra) + len(self.missing) + len(self.wrong_order)

    def get_details(self):
        return {}