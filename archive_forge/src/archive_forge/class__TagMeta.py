import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class _TagMeta(type):
    """metaclass to allow Tag to produce a subclass according to
    its keyword"""
    _classmap = {}

    def __init__(cls, clsname, bases, dict_):
        if getattr(cls, '__keyword__', None) is not None:
            cls._classmap[cls.__keyword__] = cls
        super().__init__(clsname, bases, dict_)

    def __call__(cls, keyword, attributes, **kwargs):
        if ':' in keyword:
            ns, defname = keyword.split(':')
            return type.__call__(CallNamespaceTag, ns, defname, attributes, **kwargs)
        try:
            cls = _TagMeta._classmap[keyword]
        except KeyError:
            raise exceptions.CompileException("No such tag: '%s'" % keyword, source=kwargs['source'], lineno=kwargs['lineno'], pos=kwargs['pos'], filename=kwargs['filename'])
        return type.__call__(cls, keyword, attributes, **kwargs)