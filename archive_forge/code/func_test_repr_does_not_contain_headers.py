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
def test_repr_does_not_contain_headers(self):
    request = HTTPServerRequest(uri='/', headers=HTTPHeaders({'Canary': ['Coal Mine']}))
    self.assertTrue('Canary' not in repr(request))