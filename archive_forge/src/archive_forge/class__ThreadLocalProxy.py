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
class _ThreadLocalProxy(object):
    __slots__ = ['__attrname__', '__dict__']

    def __init__(self, attrname):
        self.__attrname__ = attrname

    def __getattr__(self, name):
        child = getattr(serving, self.__attrname__)
        return getattr(child, name)

    def __setattr__(self, name, value):
        if name in ('__attrname__',):
            object.__setattr__(self, name, value)
        else:
            child = getattr(serving, self.__attrname__)
            setattr(child, name, value)

    def __delattr__(self, name):
        child = getattr(serving, self.__attrname__)
        delattr(child, name)

    @property
    def __dict__(self):
        child = getattr(serving, self.__attrname__)
        d = child.__class__.__dict__.copy()
        d.update(child.__dict__)
        return d

    def __getitem__(self, key):
        child = getattr(serving, self.__attrname__)
        return child[key]

    def __setitem__(self, key, value):
        child = getattr(serving, self.__attrname__)
        child[key] = value

    def __delitem__(self, key):
        child = getattr(serving, self.__attrname__)
        del child[key]

    def __contains__(self, key):
        child = getattr(serving, self.__attrname__)
        return key in child

    def __len__(self):
        child = getattr(serving, self.__attrname__)
        return len(child)

    def __nonzero__(self):
        child = getattr(serving, self.__attrname__)
        return bool(child)
    __bool__ = __nonzero__