import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
class Dir1:

    @cherrypy.expose
    def index(self):
        return 'index for dir1'

    @cherrypy.expose
    @cherrypy.config(**{'tools.trailing_slash.extra': True})
    def myMethod(self):
        return 'myMethod from dir1, path_info is:' + repr(cherrypy.request.path_info)

    @cherrypy.expose
    def default(self, *params):
        return 'default for dir1, param is:' + repr(params)