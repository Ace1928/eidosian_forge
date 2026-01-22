import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
class TestRequest(object):
    __test__ = False
    '\n    Instances of this class are created by `TestApp\n    <class-paste.fixture.TestApp.html>`_ with the ``.get()`` and\n    ``.post()`` methods, and are consumed there by ``.do_request()``.\n\n    Instances are also available as a ``.req`` attribute on\n    `TestResponse <class-paste.fixture.TestResponse.html>`_ instances.\n\n    Useful attributes:\n\n    ``url``:\n        The url (actually usually the path) of the request, without\n        query string.\n\n    ``environ``:\n        The environment dictionary used for the request.\n\n    ``full_url``:\n        The url/path, with query string.\n    '

    def __init__(self, url, environ, expect_errors=False):
        if url.startswith('http://localhost'):
            url = url[len('http://localhost'):]
        self.url = url
        self.environ = environ
        if environ.get('QUERY_STRING'):
            self.full_url = url + '?' + environ['QUERY_STRING']
        else:
            self.full_url = url
        self.expect_errors = expect_errors