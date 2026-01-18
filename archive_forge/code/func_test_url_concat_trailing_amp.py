from tornado.httputil import (
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List
def test_url_concat_trailing_amp(self):
    url = url_concat('https://localhost/path?x&', [('y', 'y'), ('z', 'z')])
    self.assertEqual(url, 'https://localhost/path?x=&y=y&z=z')