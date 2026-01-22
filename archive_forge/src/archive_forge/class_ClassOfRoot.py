import cherrypy
from cherrypy.test import helper
class ClassOfRoot(object):

    def __init__(self, name):
        self.name = name

    @cherrypy.expose
    def index(self):
        return 'Welcome to the %s website!' % self.name