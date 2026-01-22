from . import trace
class _CachedPropertyForAttr:

    def __init__(self, attrname):
        self.attrname = attrname

    def __call__(self, fn):
        return _CachedProperty(self.attrname, fn)