import cherrypy
from cherrypy.lib import auth_digest
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def test_wrong_scheme(self):
    basic_auth = {'Authorization': 'Basic foo:bar'}
    self.getPage('/digest/', headers=list(basic_auth.items()))
    assert self.status_code == 401