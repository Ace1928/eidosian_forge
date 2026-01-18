import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
def test_simple_batch(self):
    get_timeout = 3
    batch_timeout = 2
    if self.notify_url.startswith('amqp:'):
        backend = os.environ.get('AMQP1_BACKEND')
        if backend == 'qdrouterd':
            self.skipTest('qdrouterd backend')
    if self.notify_url.startswith('kafka://'):
        get_timeout = 10
        batch_timeout = 5
        self.conf.set_override('consumer_group', 'test_simple_batch', group='oslo_messaging_kafka')
    listener = self.useFixture(utils.BatchNotificationFixture(self.conf, self.notify_url, ['test_simple_batch'], batch_size=100, batch_timeout=batch_timeout))
    notifier = listener.notifier('abc')
    for i in range(0, 205):
        notifier.info({}, 'test%s' % i, 'Hello World!')
    events = listener.get_events(timeout=get_timeout)
    self.assertEqual(3, len(events))
    self.assertEqual(100, len(events[0][1]))
    self.assertEqual(100, len(events[1][1]))
    self.assertEqual(5, len(events[2][1]))