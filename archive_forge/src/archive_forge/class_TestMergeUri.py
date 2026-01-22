import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class TestMergeUri(test.TestCase):

    def test_merge(self):
        url = 'http://www.yahoo.com/?a=b&c=d'
        parsed = misc.parse_uri(url)
        joined = misc.merge_uri(parsed, {})
        self.assertEqual('b', joined.get('a'))
        self.assertEqual('d', joined.get('c'))
        self.assertEqual('www.yahoo.com', joined.get('hostname'))

    def test_merge_existing_hostname(self):
        url = 'http://www.yahoo.com/'
        parsed = misc.parse_uri(url)
        joined = misc.merge_uri(parsed, {'hostname': 'b.com'})
        self.assertEqual('b.com', joined.get('hostname'))

    def test_merge_user_password(self):
        url = 'http://josh:harlow@www.yahoo.com/'
        parsed = misc.parse_uri(url)
        joined = misc.merge_uri(parsed, {})
        self.assertEqual('www.yahoo.com', joined.get('hostname'))
        self.assertEqual('josh', joined.get('username'))
        self.assertEqual('harlow', joined.get('password'))

    def test_merge_user_password_existing(self):
        url = 'http://josh:harlow@www.yahoo.com/'
        parsed = misc.parse_uri(url)
        existing = {'username': 'joe', 'password': 'biggie'}
        joined = misc.merge_uri(parsed, existing)
        self.assertEqual('www.yahoo.com', joined.get('hostname'))
        self.assertEqual('joe', joined.get('username'))
        self.assertEqual('biggie', joined.get('password'))