import os
import cherrypy
from cherrypy.test import helper
@cherrypy.expose
def vmethod(self, value):
    return 'You sent %s' % value