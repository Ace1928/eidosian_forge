import cherrypy
from cherrypy.test import helper
def test_welcome(self):
    if not cherrypy.server.using_wsgi:
        return self.skip('skipped (not using WSGI)... ')
    for year in range(1997, 2008):
        self.getPage('/', headers=[('Host', 'www.classof%s.example' % year)])
        self.assertBody('Welcome to the Class of %s website!' % year)