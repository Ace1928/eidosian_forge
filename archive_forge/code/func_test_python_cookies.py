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
def test_python_cookies(self):
    """
        Test cases copied from Python's Lib/test/test_http_cookies.py
        """
    self.assertEqual(parse_cookie('chips=ahoy; vienna=finger'), {'chips': 'ahoy', 'vienna': 'finger'})
    self.assertEqual(parse_cookie('keebler="E=mc2; L=\\"Loves\\"; fudge=\\012;"'), {'keebler': '"E=mc2', 'L': '\\"Loves\\"', 'fudge': '\\012', '': '"'})
    self.assertEqual(parse_cookie('keebler=E=mc2'), {'keebler': 'E=mc2'})
    self.assertEqual(parse_cookie('key:term=value:term'), {'key:term': 'value:term'})
    self.assertEqual(parse_cookie('a=b; c=[; d=r; f=h'), {'a': 'b', 'c': '[', 'd': 'r', 'f': 'h'})