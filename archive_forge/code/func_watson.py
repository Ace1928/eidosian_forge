import cherrypy
from cherrypy import expose, tools
@expose(alias='alias3')
def watson(self):
    return 'Mr. and Mrs. Watson'