import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def _bad_response(self, method, url, body, headers):
    self._check_request(url)
    result = {'success': True}
    return self._response(httplib.OK, result, httplib.responses[httplib.OK])