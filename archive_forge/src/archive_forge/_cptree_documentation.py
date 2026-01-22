import os
import cherrypy
from cherrypy import _cpconfig, _cplogging, _cprequest, _cpwsgi, tools
from cherrypy.lib import httputil, reprconf
Pre-initialize WSGI env and call WSGI-callable.