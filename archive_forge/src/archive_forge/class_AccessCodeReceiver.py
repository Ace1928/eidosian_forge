import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
class AccessCodeReceiver(BaseHTTPRequestHandler):

    def do_GET(self_):
        query = urlparse.urlparse(self_.path).query
        query_components = dict((qc.split('=') for qc in query.split('&')))
        if 'state' in query_components and query_components['state'] != urllib.parse.quote(self._state):
            raise ValueError("States do not match: {} != {}, can't trust authentication".format(self._state, query_components['state']))
        nonlocal access_code
        access_code = query_components['code']
        self_.send_response(200)
        self_.send_header('Content-type', 'text/html')
        self_.end_headers()
        self_.wfile.write(b'<html><head><title>Libcloud Sign-In</title></head>')
        self_.wfile.write(b'<body><p>You can now close this tab</p>')