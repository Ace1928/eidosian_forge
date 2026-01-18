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
def test_parse_config_file(self):
    options = OptionParser()
    options.define('port', default=80)
    options.define('username', default='foo')
    options.define('my_path')
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'options_test.cfg')
    options.parse_config_file(config_path)
    self.assertEqual(options.port, 443)
    self.assertEqual(options.username, '李康')
    self.assertEqual(options.my_path, config_path)