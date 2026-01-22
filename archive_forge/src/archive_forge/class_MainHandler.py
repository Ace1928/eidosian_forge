import contextlib
import json
import sys
from StringIO import StringIO
import traceback
from google.appengine.api import app_identity
import google.auth
from google.auth import _helpers
from google.auth import app_engine
import google.auth.transport.urllib3
import urllib3.contrib.appengine
import webapp2
class MainHandler(webapp2.RequestHandler):

    def get(self):
        self.response.headers['content-type'] = 'text/plain'
        status, output = run_tests()
        if not status:
            self.response.status = 500
        self.response.write(output)