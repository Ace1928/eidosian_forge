import os
import os.path
import cherrypy
@cherrypy.expose
def toggleTracebacks(self):
    tracebacks = cherrypy.request.show_tracebacks
    cherrypy.config.update({'request.show_tracebacks': not tracebacks})
    raise cherrypy.HTTPRedirect('/')