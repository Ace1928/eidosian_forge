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
def _define_options(self):
    options = OptionParser()
    options.define('str', type=str)
    options.define('basestring', type=basestring_type)
    options.define('int', type=int)
    options.define('float', type=float)
    options.define('datetime', type=datetime.datetime)
    options.define('timedelta', type=datetime.timedelta)
    options.define('email', type=Email)
    options.define('list-of-int', type=int, multiple=True)
    options.define('list-of-str', type=str, multiple=True)
    return options