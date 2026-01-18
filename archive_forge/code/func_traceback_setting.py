import sys
import importlib
import cherrypy
from cherrypy.test import helper
@cherrypy.expose
def traceback_setting():
    return repr(cherrypy.request.show_tracebacks)