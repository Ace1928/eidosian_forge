import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
def test_nontext(self):
    self.getPage('/nontext')
    self.assertHeader('Content-Type', 'application/binary;charset=utf-8')
    self.assertBody('\x00\x01\x02\x03')