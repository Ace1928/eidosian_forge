import cherrypy
from cherrypy.test import helper
class NonDecoratedPopArgs:
    """Test _cp_dispatch = cherrypy.popargs()"""
    _cp_dispatch = cherrypy.popargs('a')

    @cherrypy.expose
    def index(self, a):
        return 'index: ' + str(a)