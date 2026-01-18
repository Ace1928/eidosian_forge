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
def test_utf8_logging(self):
    with ignore_bytes_warning():
        self.logger.error('é'.encode('utf8'))
    if issubclass(bytes, basestring_type):
        self.assertEqual(self.get_output(), utf8('é'))
    else:
        self.assertEqual(self.get_output(), utf8(repr(utf8('é'))))