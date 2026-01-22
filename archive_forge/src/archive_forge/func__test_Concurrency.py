import os
import platform
import threading
import time
from http.client import HTTPConnection
from distutils.spawn import find_executable
import pytest
from path import Path
from more_itertools import consume
import portend
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.lib import sessions
from cherrypy.lib import reprconf
from cherrypy.lib.httputil import response_codes
from cherrypy.test import helper
from cherrypy import _json as json
def _test_Concurrency(self):
    client_thread_count = 5
    request_count = 30
    self.getPage('/')
    assert self.body == b'1'
    cookies = self.cookies
    data_dict = {}
    errors = []

    def request(index):
        if self.scheme == 'https':
            c = HTTPSConnection('%s:%s' % (self.interface(), self.PORT))
        else:
            c = HTTPConnection('%s:%s' % (self.interface(), self.PORT))
        for i in range(request_count):
            c.putrequest('GET', '/')
            for k, v in cookies:
                c.putheader(k, v)
            c.endheaders()
            response = c.getresponse()
            body = response.read()
            if response.status != 200 or not body.isdigit():
                errors.append((response.status, body))
            else:
                data_dict[index] = max(data_dict[index], int(body))
    ts = []
    for c in range(client_thread_count):
        data_dict[c] = 0
        t = threading.Thread(target=request, args=(c,))
        ts.append(t)
        t.start()
    for t in ts:
        t.join()
    hitcount = max(data_dict.values())
    expected = 1 + client_thread_count * request_count
    for e in errors:
        print(e)
    assert len(errors) == 0
    assert hitcount == expected