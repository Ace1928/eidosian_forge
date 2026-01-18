import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
def test_redir_using_url(self):
    for url in script_names:
        self.script_name = url
        self.getPage('/redirect_via_url?path=./')
        self.assertStatus(('302 Found', '303 See Other'))
        self.assertHeader('Location', '%s/' % self.base())
        self.getPage('/redirect_via_url?path=./')
        self.assertStatus(('302 Found', '303 See Other'))
        self.assertHeader('Location', '%s/' % self.base())
        self.getPage('/redirect_via_url/?path=./')
        self.assertStatus(('302 Found', '303 See Other'))
        self.assertHeader('Location', '%s/' % self.base())
        self.getPage('/redirect_via_url/?path=./')
        self.assertStatus(('302 Found', '303 See Other'))
        self.assertHeader('Location', '%s/' % self.base())