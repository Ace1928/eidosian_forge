import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def test_start_response_error(self):
    self.getPage('/start_response_error')
    self.assertStatus(500)
    self.assertInBody('TypeError: response.header_list key 2 is not a byte string.')