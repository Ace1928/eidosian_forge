import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
def _setargs(self):
    """Copy func parameter names to obj attributes."""
    try:
        for arg in _getargs(self.callable):
            setattr(self, arg, None)
    except (TypeError, AttributeError):
        if hasattr(self.callable, '__call__'):
            for arg in _getargs(self.callable.__call__):
                setattr(self, arg, None)
    except NotImplementedError:
        pass
    except IndexError:
        pass