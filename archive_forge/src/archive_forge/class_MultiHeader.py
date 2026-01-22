import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
class MultiHeader(Test):

    def header_list(self):
        pass
    header_list = cherrypy.tools.append_headers(header_list=[(b'WWW-Authenticate', b'Negotiate'), (b'WWW-Authenticate', b'Basic realm="foo"')])(header_list)

    def commas(self):
        cherrypy.response.headers['WWW-Authenticate'] = 'Negotiate,Basic realm="foo"'