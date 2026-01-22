import cherrypy
from cherrypy import expose, tools
@expose(['alias1', 'alias2'])
def andrews(self):
    return 'Mr Ken Andrews'