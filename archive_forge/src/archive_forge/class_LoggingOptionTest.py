import contextlib
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from tornado.escape import utf8
from tornado.log import LogFormatter, define_logging_options, enable_pretty_logging
from tornado.options import OptionParser
from tornado.util import basestring_type
class LoggingOptionTest(unittest.TestCase):
    """Test the ability to enable and disable Tornado's logging hooks."""

    def logs_present(self, statement, args=None):
        IMPORT = 'from tornado.options import options, parse_command_line'
        LOG_INFO = 'import logging; logging.info("hello")'
        program = ';'.join([IMPORT, statement, LOG_INFO])
        proc = subprocess.Popen([sys.executable, '-c', program] + (args or []), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = proc.communicate()
        self.assertEqual(proc.returncode, 0, 'process failed: %r' % stdout)
        return b'hello' in stdout

    def test_default(self):
        self.assertFalse(self.logs_present('pass'))

    def test_tornado_default(self):
        self.assertTrue(self.logs_present('parse_command_line()'))

    def test_disable_command_line(self):
        self.assertFalse(self.logs_present('parse_command_line()', ['--logging=none']))

    def test_disable_command_line_case_insensitive(self):
        self.assertFalse(self.logs_present('parse_command_line()', ['--logging=None']))

    def test_disable_code_string(self):
        self.assertFalse(self.logs_present('options.logging = "none"; parse_command_line()'))

    def test_disable_code_none(self):
        self.assertFalse(self.logs_present('options.logging = None; parse_command_line()'))

    def test_disable_override(self):
        self.assertTrue(self.logs_present('options.logging = None; parse_command_line()', ['--logging=info']))