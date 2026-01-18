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
def test_subcommand(self):
    base_options = OptionParser()
    base_options.define('verbose', default=False)
    sub_options = OptionParser()
    sub_options.define('foo', type=str)
    rest = base_options.parse_command_line(['main.py', '--verbose', 'subcommand', '--foo=bar'])
    self.assertEqual(rest, ['subcommand', '--foo=bar'])
    self.assertTrue(base_options.verbose)
    rest2 = sub_options.parse_command_line(rest)
    self.assertEqual(rest2, [])
    self.assertEqual(sub_options.foo, 'bar')
    try:
        orig_stderr = sys.stderr
        sys.stderr = StringIO()
        with self.assertRaises(Error):
            sub_options.parse_command_line(['subcommand', '--verbose'])
    finally:
        sys.stderr = orig_stderr