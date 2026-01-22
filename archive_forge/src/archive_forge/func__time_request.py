import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def _time_request(self, url, method, **kwargs):
    start_time = time.time()
    resp, body = self.request(url, method, **kwargs)
    self.times.append(('%s %s' % (method, url), start_time, time.time()))
    return (resp, body)