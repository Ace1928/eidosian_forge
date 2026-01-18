from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def test_repeated_headers(self):
    self.getPage('/headers/Accept-Charset', headers=[('Accept-Charset', 'iso-8859-5'), ('Accept-Charset', 'unicode-1-1;q=0.8')])
    self.assertBody('iso-8859-5, unicode-1-1;q=0.8')
    self.getPage('/headers/doubledheaders')
    self.assertBody('double header test')
    hnames = [name.title() for name, val in self.headers]
    for key in ['Content-Length', 'Content-Type', 'Date', 'Expires', 'Location', 'Server']:
        self.assertEqual(hnames.count(key), 1, self.headers)