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
@pytest.mark.skipif(platform.system() != 'Windows', reason='Windows only')
def test_static_longpath(self):
    """Test serving of a file in subdir of a Windows long-path
        staticdir."""
    self.getPage('/static-long/static/index.html')
    self.assertStatus('200 OK')
    self.assertHeader('Content-Type', 'text/html')
    self.assertBody('Hello, world\r\n')