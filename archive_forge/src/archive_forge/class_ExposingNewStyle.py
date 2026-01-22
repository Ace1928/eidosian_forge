import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
class ExposingNewStyle(object):

    @cherrypy.expose
    def base(self):
        return 'expose works!'
    cherrypy.expose(base, '1')
    cherrypy.expose(base, '2')