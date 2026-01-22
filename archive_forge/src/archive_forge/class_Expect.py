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
class Expect(Test):

    def expectation_failed(self):
        expect = cherrypy.request.headers.elements('Expect')
        if expect and expect[0].value != '100-continue':
            raise cherrypy.HTTPError(400)
        raise cherrypy.HTTPError(417, 'Expectation Failed')