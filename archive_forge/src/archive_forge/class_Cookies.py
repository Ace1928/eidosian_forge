import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
class Cookies(Test):

    def single(self, name):
        cookie = cherrypy.request.cookie[name]
        cherrypy.response.cookie[str(name)] = cookie.value

    def multiple(self, names):
        list(map(self.single, names))