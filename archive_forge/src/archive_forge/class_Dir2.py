import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
class Dir2:

    @cherrypy.expose
    def index(self):
        return 'index for dir2, path is:' + cherrypy.request.path_info

    @cherrypy.expose
    def script_name(self):
        return cherrypy.tree.script_name()

    @cherrypy.expose
    def cherrypy_url(self):
        return cherrypy.url('/extra')

    @cherrypy.expose
    def posparam(self, *vpath):
        return '/'.join(vpath)