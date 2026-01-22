from threading import local as _local
from ._cperror import (
from . import _cpdispatch as dispatch
from ._cptools import default_toolbox as tools, Tool
from ._helper import expose, popargs, url
from . import _cprequest, _cpserver, _cptree, _cplogging, _cpconfig
import cherrypy.lib.httputil as _httputil
from ._cptree import Application
from . import _cpwsgi as wsgi
from . import process
from . import _cpchecker
class _GlobalLogManager(_cplogging.LogManager):
    """A site-wide LogManager; routes to app.log or global log as appropriate.

    This :class:`LogManager<cherrypy._cplogging.LogManager>` implements
    cherrypy.log() and cherrypy.log.access(). If either
    function is called during a request, the message will be sent to the
    logger for the current Application. If they are called outside of a
    request, the message will be sent to the site-wide logger.
    """

    def __call__(self, *args, **kwargs):
        """Log the given message to the app.log or global log.

        Log the given message to the app.log or global
        log as appropriate.
        """
        if hasattr(request, 'app') and hasattr(request.app, 'log'):
            log = request.app.log
        else:
            log = self
        return log.error(*args, **kwargs)

    def access(self):
        """Log an access message to the app.log or global log.

        Log the given message to the app.log or global
        log as appropriate.
        """
        try:
            return request.app.log.access()
        except AttributeError:
            return _cplogging.LogManager.access(self)