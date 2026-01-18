import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def url_with_xss(self):
    raise cherrypy.HTTPRedirect("/some<script>alert(1);</script>url/that'we/want")