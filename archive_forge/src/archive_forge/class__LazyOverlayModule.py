import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
class _LazyOverlayModule(ModuleType):
    """proxy module which overlays original module,
    and lazily imports specified attributes.

    this is mainly used to prevent importing of resources
    that are only needed by certain password hashes,
    yet allow them to be imported from a single location.

    used by :mod:`passlib.utils`, :mod:`passlib.crypto`,
    and :mod:`passlib.utils.compat`.
    """

    @classmethod
    def replace_module(cls, name, attrmap):
        orig = sys.modules[name]
        self = cls(name, attrmap, orig)
        sys.modules[name] = self
        return self

    def __init__(self, name, attrmap, proxy=None):
        ModuleType.__init__(self, name)
        self.__attrmap = attrmap
        self.__proxy = proxy
        self.__log = logging.getLogger(name)

    def __getattr__(self, attr):
        proxy = self.__proxy
        if proxy and hasattr(proxy, attr):
            return getattr(proxy, attr)
        attrmap = self.__attrmap
        if attr in attrmap:
            source = attrmap[attr]
            if callable(source):
                value = source()
            else:
                value = _import_object(source)
            setattr(self, attr, value)
            self.__log.debug('loaded lazy attr %r: %r', attr, value)
            return value
        raise AttributeError("'module' object has no attribute '%s'" % (attr,))

    def __repr__(self):
        proxy = self.__proxy
        if proxy:
            return repr(proxy)
        else:
            return ModuleType.__repr__(self)

    def __dir__(self):
        attrs = set(dir(self.__class__))
        attrs.update(self.__dict__)
        attrs.update(self.__attrmap)
        proxy = self.__proxy
        if proxy is not None:
            attrs.update(dir(proxy))
        return list(attrs)