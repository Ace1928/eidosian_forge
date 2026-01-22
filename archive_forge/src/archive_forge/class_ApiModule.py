import os
import sys
from types import ModuleType
from .version import version as __version__  # NOQA:F401
class ApiModule(ModuleType):
    """the magical lazy-loading module standing"""

    def __docget(self):
        try:
            return self.__doc
        except AttributeError:
            if '__doc__' in self.__map__:
                return self.__makeattr('__doc__')

    def __docset(self, value):
        self.__doc = value
    __doc__ = property(__docget, __docset)

    def __init__(self, name, importspec, implprefix=None, attr=None):
        self.__name__ = name
        self.__all__ = [x for x in importspec if x != '__onfirstaccess__']
        self.__map__ = {}
        self.__implprefix__ = implprefix or name
        if attr:
            for name, val in attr.items():
                setattr(self, name, val)
        for name, importspec in importspec.items():
            if isinstance(importspec, dict):
                subname = '{}.{}'.format(self.__name__, name)
                apimod = ApiModule(subname, importspec, implprefix)
                sys.modules[subname] = apimod
                setattr(self, name, apimod)
            else:
                parts = importspec.split(':')
                modpath = parts.pop(0)
                attrname = parts and parts[0] or ''
                if modpath[0] == '.':
                    modpath = implprefix + modpath
                if not attrname:
                    subname = '{}.{}'.format(self.__name__, name)
                    apimod = AliasModule(subname, modpath)
                    sys.modules[subname] = apimod
                    if '.' not in name:
                        setattr(self, name, apimod)
                else:
                    self.__map__[name] = (modpath, attrname)

    def __repr__(self):
        repr_list = []
        if hasattr(self, '__version__'):
            repr_list.append('version=' + repr(self.__version__))
        if hasattr(self, '__file__'):
            repr_list.append('from ' + repr(self.__file__))
        if repr_list:
            return '<ApiModule {!r} {}>'.format(self.__name__, ' '.join(repr_list))
        return '<ApiModule {!r}>'.format(self.__name__)

    def __makeattr(self, name):
        """lazily compute value for name or raise AttributeError if unknown."""
        target = None
        if '__onfirstaccess__' in self.__map__:
            target = self.__map__.pop('__onfirstaccess__')
            importobj(*target)()
        try:
            modpath, attrname = self.__map__[name]
        except KeyError:
            if target is not None and name != '__onfirstaccess__':
                return getattr(self, name)
            raise AttributeError(name)
        else:
            result = importobj(modpath, attrname)
            setattr(self, name, result)
            try:
                del self.__map__[name]
            except KeyError:
                pass
            return result
    __getattr__ = __makeattr

    @property
    def __dict__(self):
        dictdescr = ModuleType.__dict__['__dict__']
        dict = dictdescr.__get__(self)
        if dict is not None:
            hasattr(self, 'some')
            for name in self.__all__:
                try:
                    self.__makeattr(name)
                except AttributeError:
                    pass
        return dict