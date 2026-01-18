import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_port_provided(self):
    url = 'rabbitmq://www.yahoo.com:5672'
    parsed = misc.parse_uri(url)
    self.assertEqual('rabbitmq', parsed.scheme)
    self.assertEqual('www.yahoo.com', parsed.hostname)
    self.assertEqual(5672, parsed.port)
    self.assertEqual('', parsed.path)