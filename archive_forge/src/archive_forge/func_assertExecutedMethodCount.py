import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
def assertExecutedMethodCount(self, expected):
    actual = len(self._executed_mock_methods)
    self.assertEqual(actual, expected, 'expected %d, but %d mock methods were executed' % (expected, actual))