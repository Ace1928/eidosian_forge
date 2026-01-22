import datetime
from io import StringIO
import os
import sys
from unittest import mock
import unittest
from tornado.options import OptionParser, Error
from tornado.util import basestring_type
from tornado.test.util import subTest
import typing
def _check_options_values(self, options):
    self.assertEqual(options.str, 'asdf')
    self.assertEqual(options.basestring, 'qwer')
    self.assertEqual(options.int, 42)
    self.assertEqual(options.float, 1.5)
    self.assertEqual(options.datetime, datetime.datetime(2013, 4, 28, 5, 16))
    self.assertEqual(options.timedelta, datetime.timedelta(seconds=45))
    self.assertEqual(options.email.value, 'tornado@web.com')
    self.assertTrue(isinstance(options.email, Email))
    self.assertEqual(options.list_of_int, [1, 2, 3])
    self.assertEqual(options.list_of_str, ['a', 'b', 'c'])