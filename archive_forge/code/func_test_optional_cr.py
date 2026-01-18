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
def test_optional_cr(self):
    headers = HTTPHeaders.parse('CRLF: crlf\r\nLF: lf\nCR: cr\rMore: more\r\n')
    self.assertEqual(sorted(headers.get_all()), [('Cr', 'cr\rMore: more'), ('Crlf', 'crlf'), ('Lf', 'lf')])