import os.path
import cherrypy
class ExtraLinksPage:

    @cherrypy.expose
    def index(self):
        return '\n            <p>Here are some extra useful links:</p>\n\n            <ul>\n                <li><a href="http://del.icio.us">del.icio.us</a></li>\n                <li><a href="http://www.cherrypy.dev">CherryPy</a></li>\n            </ul>\n\n            <p>[<a href="../">Return to links page</a>]</p>'