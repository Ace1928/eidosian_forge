import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class UriParseTest(test.TestCase):

    def test_parse(self):
        url = 'zookeeper://192.168.0.1:2181/a/b/?c=d'
        parsed = misc.parse_uri(url)
        self.assertEqual('zookeeper', parsed.scheme)
        self.assertEqual(2181, parsed.port)
        self.assertEqual('192.168.0.1', parsed.hostname)
        self.assertEqual('', parsed.fragment)
        self.assertEqual('/a/b/', parsed.path)
        self.assertEqual({'c': 'd'}, parsed.params())

    def test_port_provided(self):
        url = 'rabbitmq://www.yahoo.com:5672'
        parsed = misc.parse_uri(url)
        self.assertEqual('rabbitmq', parsed.scheme)
        self.assertEqual('www.yahoo.com', parsed.hostname)
        self.assertEqual(5672, parsed.port)
        self.assertEqual('', parsed.path)

    def test_ipv6_host(self):
        url = 'rsync://[2001:db8:0:1::2]:873'
        parsed = misc.parse_uri(url)
        self.assertEqual('rsync', parsed.scheme)
        self.assertEqual('2001:db8:0:1::2', parsed.hostname)
        self.assertEqual(873, parsed.port)

    def test_user_password(self):
        url = 'rsync://test:test_pw@www.yahoo.com:873'
        parsed = misc.parse_uri(url)
        self.assertEqual('test', parsed.username)
        self.assertEqual('test_pw', parsed.password)
        self.assertEqual('www.yahoo.com', parsed.hostname)

    def test_user(self):
        url = 'rsync://test@www.yahoo.com:873'
        parsed = misc.parse_uri(url)
        self.assertEqual('test', parsed.username)
        self.assertIsNone(parsed.password)