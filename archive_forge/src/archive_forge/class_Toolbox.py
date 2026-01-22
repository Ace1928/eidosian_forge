import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
class Toolbox(object):
    """A collection of Tools.

    This object also functions as a config namespace handler for itself.
    Custom toolboxes should be added to each Application's toolboxes dict.
    """

    def __init__(self, namespace):
        self.namespace = namespace

    def __setattr__(self, name, value):
        if isinstance(value, Tool):
            if value._name is None:
                value._name = name
            value.namespace = self.namespace
        object.__setattr__(self, name, value)

    def __enter__(self):
        """Populate request.toolmaps from tools specified in config."""
        cherrypy.serving.request.toolmaps[self.namespace] = map = {}

        def populate(k, v):
            toolname, arg = k.split('.', 1)
            bucket = map.setdefault(toolname, {})
            bucket[arg] = v
        return populate

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Run tool._setup() for each tool in our toolmap."""
        map = cherrypy.serving.request.toolmaps.get(self.namespace)
        if map:
            for name, settings in map.items():
                if settings.get('on', False):
                    tool = getattr(self, name)
                    tool._setup()

    def register(self, point, **kwargs):
        """
        Return a decorator which registers the function
        at the given hook point.
        """

        def decorator(func):
            attr_name = kwargs.get('name', func.__name__)
            tool = Tool(point, func, **kwargs)
            setattr(self, attr_name, tool)
            return func
        return decorator