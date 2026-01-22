import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def append_headers(header_list, debug=False):
    if debug:
        cherrypy.log('Extending response headers with %s' % repr(header_list), 'TOOLS.APPEND_HEADERS')
    cherrypy.serving.response.header_list.extend(header_list)