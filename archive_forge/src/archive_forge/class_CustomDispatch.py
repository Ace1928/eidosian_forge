import cherrypy
from cherrypy.test import helper
class CustomDispatch:

    @cherrypy.expose
    def index(self, a, b):
        return 'custom'