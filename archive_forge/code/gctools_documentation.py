import gc
import inspect
import sys
import time
import cherrypy
from cherrypy import _cprequest, _cpwsgi
from cherrypy.process.plugins import SimplePlugin
Return a list of string reprs from a nested list of referrers.