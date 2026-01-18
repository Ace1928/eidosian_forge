import io
import os
import sys
import re
import platform
import tempfile
import urllib.parse
import unittest.mock
from http.client import HTTPConnection
import pytest
import py.path
import path
import cherrypy
from cherrypy.lib import static
from cherrypy._cpcompat import HTTPSConnection, ntou, tonative
from cherrypy.test import helper
def test_serve_fileobj(self):
    self.getPage('/fileobj')
    self.assertStatus('200 OK')
    self.assertHeader('Content-Type', 'text/css;charset=utf-8')
    self.assertMatchesBody('^Dummy stylesheet')