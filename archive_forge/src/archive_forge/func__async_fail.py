import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def _async_fail(self, method, url, body, headers):
    query = self._check_request(url)
    if query['command'].lower() == 'queryasyncjobresult':
        self.assertEqual(query['jobid'], '42')
        result = {query['command'].lower() + 'response': {'jobstatus': 2, 'jobresult': {'errortext': self.ERROR_TEXT}}}
    else:
        result = {query['command'].lower() + 'response': {'jobid': '42'}}
    return self._response(httplib.OK, result, httplib.responses[httplib.OK])