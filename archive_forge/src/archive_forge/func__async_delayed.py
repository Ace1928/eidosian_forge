import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def _async_delayed(self, method, url, body, headers):
    global async_delay
    query = self._check_request(url)
    if query['command'].lower() == 'queryasyncjobresult':
        self.assertEqual(query['jobid'], '42')
        if async_delay == 0:
            result = {query['command'].lower() + 'response': {'jobstatus': 1, 'jobresult': {'fake': 'result'}}}
        else:
            result = {query['command'].lower() + 'response': {'jobstatus': 0}}
            async_delay -= 1
    else:
        result = {query['command'].lower() + 'response': {'jobid': '42'}}
    return self._response(httplib.OK, result, httplib.responses[httplib.OK])