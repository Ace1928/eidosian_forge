import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
class ResponseHeadersTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def testResponseHeadersDecorator(self):
        self.getPage('/')
        self.assertHeader('Content-Language', 'en-GB')
        self.assertHeader('Content-Type', 'text/plain;charset=utf-8')

    def testResponseHeaders(self):
        self.getPage('/other')
        self.assertHeader('Content-Language', 'fr')
        self.assertHeader('Content-Type', 'text/plain;charset=utf-8')