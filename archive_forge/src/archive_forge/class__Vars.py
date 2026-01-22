import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import reprconf
class _Vars(object):
    """Adapter allowing setting a default attribute on a function or class."""

    def __init__(self, target):
        self.target = target

    def setdefault(self, key, default):
        if not hasattr(self.target, key):
            setattr(self.target, key, default)
        return getattr(self.target, key)