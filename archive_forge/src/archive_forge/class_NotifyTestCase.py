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
class NotifyTestCase(utils.SkipIfNoTransportURL):

    def test_simple(self):
        get_timeout = 1
        if self.notify_url.startswith('kafka://'):
            get_timeout = 5
            self.conf.set_override('consumer_group', 'test_simple', group='oslo_messaging_kafka')
        listener = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['test_simple']))
        notifier = listener.notifier('abc')
        notifier.info({}, 'test', 'Hello World!')
        event = listener.events.get(timeout=get_timeout)
        self.assertEqual('info', event[0])
        self.assertEqual('test', event[1])
        self.assertEqual('Hello World!', event[2])
        self.assertEqual('abc', event[3])

    def test_multiple_topics(self):
        get_timeout = 1
        if self.notify_url.startswith('kafka://'):
            get_timeout = 5
            self.conf.set_override('consumer_group', 'test_multiple_topics', group='oslo_messaging_kafka')
        listener = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['a', 'b']))
        a = listener.notifier('pub-a', topics=['a'])
        b = listener.notifier('pub-b', topics=['b'])
        sent = {'pub-a': [a, 'test-a', 'payload-a'], 'pub-b': [b, 'test-b', 'payload-b']}
        for e in sent.values():
            e[0].info({}, e[1], e[2])
        received = {}
        while len(received) < len(sent):
            e = listener.events.get(timeout=get_timeout)
            received[e[3]] = e
        for key in received:
            actual = received[key]
            expected = sent[key]
            self.assertEqual('info', actual[0])
            self.assertEqual(expected[1], actual[1])
            self.assertEqual(expected[2], actual[2])

    def test_multiple_servers(self):
        timeout = 0.5
        if self.notify_url.startswith('amqp:'):
            self.skipTest('QPID-6307')
        if self.notify_url.startswith('kafka://'):
            self.skipTest('Kafka: needs to be fixed')
            timeout = 5
            self.conf.set_override('consumer_group', 'test_multiple_servers', group='oslo_messaging_kafka')
        listener_a = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['test-topic']))
        listener_b = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['test-topic']))
        n = listener_a.notifier('pub')
        events_out = [('test-%s' % c, 'payload-%s' % c) for c in 'abcdefgh']
        for event_type, payload in events_out:
            n.info({}, event_type, payload)
        events_in = [[(e[1], e[2]) for e in listener_a.get_events(timeout)], [(e[1], e[2]) for e in listener_b.get_events(timeout)]]
        self.assertThat(events_in, utils.IsValidDistributionOf(events_out))
        for stream in events_in:
            self.assertThat(len(stream), matchers.GreaterThan(0))

    def test_independent_topics(self):
        get_timeout = 0.5
        if self.notify_url.startswith('kafka://'):
            get_timeout = 5
            self.conf.set_override('consumer_group', 'test_independent_topics_a', group='oslo_messaging_kafka')
        listener_a = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['1']))
        if self.notify_url.startswith('kafka://'):
            self.conf.set_override('consumer_group', 'test_independent_topics_b', group='oslo_messaging_kafka')
        listener_b = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['2']))
        a = listener_a.notifier('pub-1', topics=['1'])
        b = listener_b.notifier('pub-2', topics=['2'])
        a_out = [('test-1-%s' % c, 'payload-1-%s' % c) for c in 'abcdefgh']
        for event_type, payload in a_out:
            a.info({}, event_type, payload)
        b_out = [('test-2-%s' % c, 'payload-2-%s' % c) for c in 'ijklmnop']
        for event_type, payload in b_out:
            b.info({}, event_type, payload)

        def check_received(listener, publisher, messages):
            actuals = sorted([listener.events.get(timeout=get_timeout) for __ in range(len(a_out))])
            expected = sorted([['info', m[0], m[1], publisher] for m in messages])
            self.assertEqual(expected, actuals)
        check_received(listener_a, 'pub-1', a_out)
        check_received(listener_b, 'pub-2', b_out)

    def test_all_categories(self):
        get_timeout = 1
        if self.notify_url.startswith('kafka://'):
            get_timeout = 5
            self.conf.set_override('consumer_group', 'test_all_categories', group='oslo_messaging_kafka')
        listener = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['test_all_categories']))
        n = listener.notifier('abc')
        cats = ['debug', 'audit', 'info', 'warn', 'error', 'critical']
        events = [(getattr(n, c), c, 'type-' + c, c + '-data') for c in cats]
        for e in events:
            e[0]({}, e[2], e[3])
        received = {}
        for expected in events:
            e = listener.events.get(timeout=get_timeout)
            received[e[0]] = e
        for expected in events:
            actual = received[expected[1]]
            self.assertEqual(expected[1], actual[0])
            self.assertEqual(expected[2], actual[1])
            self.assertEqual(expected[3], actual[2])

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

    def test_compression(self):
        get_timeout = 1
        if self.notify_url.startswith('amqp:'):
            self.conf.set_override('kombu_compression', 'gzip', group='oslo_messaging_rabbit')
        if self.notify_url.startswith('kafka://'):
            get_timeout = 5
            self.conf.set_override('compression_codec', 'gzip', group='oslo_messaging_kafka')
            self.conf.set_override('consumer_group', 'test_compression', group='oslo_messaging_kafka')
        listener = self.useFixture(utils.NotificationFixture(self.conf, self.notify_url, ['test_compression']))
        notifier = listener.notifier('abc')
        notifier.info({}, 'test', 'Hello World!')
        event = listener.events.get(timeout=get_timeout)
        self.assertEqual('info', event[0])
        self.assertEqual('test', event[1])
        self.assertEqual('Hello World!', event[2])
        self.assertEqual('abc', event[3])