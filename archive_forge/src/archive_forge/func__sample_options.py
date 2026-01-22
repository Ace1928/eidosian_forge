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
def _sample_options(self):
    options = OptionParser()
    options.define('a', default=1)
    options.define('b', default=2)
    return options