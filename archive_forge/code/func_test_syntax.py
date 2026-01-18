import sys
import textwrap
import cherrypy
from cherrypy.test import helper
def test_syntax(self):
    if sys.version_info < (3,):
        return self.skip('skipped (Python 3 only)')
    code = textwrap.dedent("\n            class Root:\n                @cherrypy.expose\n                @cherrypy.tools.params()\n                def resource(self, limit: int):\n                    return type(limit).__name__\n            conf = {'/': {'tools.params.on': True}}\n            cherrypy.tree.mount(Root(), config=conf)\n            ")
    exec(code)
    self.getPage('/resource?limit=0')
    self.assertStatus(200)
    self.assertBody('int')