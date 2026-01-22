import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
@cherrypy.config(**{'tools.accept.on': True})
class Accept:

    @cherrypy.expose
    def index(self):
        return '<a href="feed">Atom feed</a>'

    @cherrypy.expose
    @tools.accept(media='application/atom+xml')
    def feed(self):
        return '<?xml version="1.0" encoding="utf-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">\n    <title>Unknown Blog</title>\n</feed>'

    @cherrypy.expose
    def select(self):
        mtype = tools.accept.callable(['text/html', 'text/plain'])
        if mtype == 'text/html':
            return '<h2>Page Title</h2>'
        else:
            return 'PAGE TITLE'