import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
class CachingTool(Tool):
    """Caching Tool for CherryPy."""

    def _wrapper(self, **kwargs):
        request = cherrypy.serving.request
        if _caching.get(**kwargs):
            request.handler = None
        elif request.cacheable:
            request.hooks.attach('before_finalize', _caching.tee_output, priority=100)
    _wrapper.priority = 90

    def _setup(self):
        """Hook caching into cherrypy.request."""
        conf = self._merged_args()
        p = conf.pop('priority', None)
        cherrypy.serving.request.hooks.attach('before_handler', self._wrapper, priority=p, **conf)