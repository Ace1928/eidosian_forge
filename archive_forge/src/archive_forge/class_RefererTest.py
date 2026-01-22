import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
class RefererTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def testReferer(self):
        self.getPage('/referer/accept')
        self.assertErrorPage(403, 'Forbidden Referer header.')
        self.getPage('/referer/accept', headers=[('Referer', 'http://www.example.com/')])
        self.assertStatus(200)
        self.assertBody('Accepted!')
        self.getPage('/referer/reject')
        self.assertStatus(200)
        self.assertBody('Accepted!')
        self.getPage('/referer/reject', headers=[('Referer', 'http://www.example.com/')])
        self.assertErrorPage(403, 'Forbidden Referer header.')