import gzip
import io
import sys
import time
import types
import unittest
import operator
from http.client import IncompleteRead
import cherrypy
from cherrypy import tools
from cherrypy._cpcompat import ntou
from cherrypy.test import helper, _test_decorators
class SessionAuthTest(unittest.TestCase):

    def test_login_screen_returns_bytes(self):
        """
        login_screen must return bytes even if unicode parameters are passed.
        Issue 1132 revealed that login_screen would return unicode if the
        username and password were unicode.
        """
        sa = cherrypy.lib.cptools.SessionAuth()
        res = sa.login_screen(None, username=str('nobody'), password=str('anypass'))
        self.assertTrue(isinstance(res, bytes))