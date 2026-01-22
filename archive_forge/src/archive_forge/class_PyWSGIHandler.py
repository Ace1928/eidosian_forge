import os
import sys
from datetime import datetime
from functools import partial
import time
from gevent.pool import Pool
from gevent.server import StreamServer
from gevent import hub, monkey, socket, pywsgi
import gunicorn
from gunicorn.http.wsgi import base_environ
from gunicorn.sock import ssl_context
from gunicorn.workers.base_async import AsyncWorker
class PyWSGIHandler(pywsgi.WSGIHandler):

    def log_request(self):
        start = datetime.fromtimestamp(self.time_start)
        finish = datetime.fromtimestamp(self.time_finish)
        response_time = finish - start
        resp_headers = getattr(self, 'response_headers', {})
        resp = GeventResponse(self.status, resp_headers, self.response_length)
        if hasattr(self, 'headers'):
            req_headers = self.headers.items()
        else:
            req_headers = []
        self.server.log.access(resp, req_headers, self.environ, response_time)

    def get_environ(self):
        env = super().get_environ()
        env['gunicorn.sock'] = self.socket
        env['RAW_URI'] = self.path
        return env