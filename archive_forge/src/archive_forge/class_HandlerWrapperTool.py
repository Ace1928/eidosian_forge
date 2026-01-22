import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
class HandlerWrapperTool(Tool):
    """Tool which wraps request.handler in a provided wrapper function.

    The 'newhandler' arg must be a handler wrapper function that takes a
    'next_handler' argument, plus ``*args`` and ``**kwargs``. Like all
    page handler
    functions, it must return an iterable for use as cherrypy.response.body.

    For example, to allow your 'inner' page handlers to return dicts
    which then get interpolated into a template::

        def interpolator(next_handler, *args, **kwargs):
            filename = cherrypy.request.config.get('template')
            cherrypy.response.template = env.get_template(filename)
            response_dict = next_handler(*args, **kwargs)
            return cherrypy.response.template.render(**response_dict)
        cherrypy.tools.jinja = HandlerWrapperTool(interpolator)
    """

    def __init__(self, newhandler, point='before_handler', name=None, priority=50):
        self.newhandler = newhandler
        self._point = point
        self._name = name
        self._priority = priority

    def callable(self, *args, **kwargs):
        innerfunc = cherrypy.serving.request.handler

        def wrap(*args, **kwargs):
            return self.newhandler(innerfunc, *args, **kwargs)
        cherrypy.serving.request.handler = wrap