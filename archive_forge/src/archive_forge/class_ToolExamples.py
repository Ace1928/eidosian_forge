import cherrypy
from cherrypy import expose, tools
class ToolExamples(object):

    @expose
    @cherrypy.config(**{'response.stream': True})
    @tools.response_headers(headers=[('Content-Type', 'application/data')])
    def blah(self):
        yield b'blah'