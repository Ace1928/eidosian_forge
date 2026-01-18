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
def test_parse_callbacks(self):
    options = OptionParser()
    self.called = False

    def callback():
        self.called = True
    options.add_parse_callback(callback)
    options.parse_command_line(['main.py'], final=False)
    self.assertFalse(self.called)
    options.parse_command_line(['main.py'])
    self.assertTrue(self.called)
    self.called = False
    options.parse_command_line(['main.py'])
    self.assertTrue(self.called)